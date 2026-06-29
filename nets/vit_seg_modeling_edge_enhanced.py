from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import copy
import logging
import math
import numpy as np

from os.path import join as pjoin

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.nn import CrossEntropyLoss, Dropout, Softmax, Linear, Conv2d, LayerNorm
from torch.nn.modules.utils import _pair
from scipy import ndimage
from . import vit_seg_configs as configs
from .vit_seg_modeling_resnet_skip import ResNetV2
from .edge.edge_enhancer import EdgePriorModule, InteractionBlock, deform_inputs, BasicBlock

logger = logging.getLogger(__name__)

ATTENTION_Q = "MultiHeadDotProductAttention_1/query"
ATTENTION_K = "MultiHeadDotProductAttention_1/key"
ATTENTION_V = "MultiHeadDotProductAttention_1/value"
ATTENTION_OUT = "MultiHeadDotProductAttention_1/out"
FC_0 = "MlpBlock_3/Dense_0"
FC_1 = "MlpBlock_3/Dense_1"
ATTENTION_NORM = "LayerNorm_0"
MLP_NORM = "LayerNorm_2"

def np2th(weights, conv=False):
    """Possibly convert HWIO to OIHW."""
    if conv:
        weights = weights.transpose([3, 2, 0, 1])
    return torch.from_numpy(weights)

def swish(x):
    return x * torch.sigmoid(x)

ACT2FN = {"gelu": torch.nn.functional.gelu, "relu": torch.nn.functional.relu, "swish": swish}


class PairShuffleMod(nn.Module):
    """
    Internal module to handle Pair Shuffle and Restoration within the model.
    Handles both Image tensors (B, C, H, W) and Token tensors (B, N, D).
    """
    def __init__(self, n_split=2):
        super(PairShuffleMod, self).__init__()
        self.n_split = n_split

    def shuffle(self, x):
        """
        Shuffles input image tensor.
        Args:
            x: [B, C, H, W]
        Returns:
            mixed_img: [B, C, H, W]
            indices: [B, Total_Patches] (Transformation indices)
        """
        B, C, H, W = x.shape

        if H % self.n_split != 0 or W % self.n_split != 0:
            return x, None # Fail safe
        
        h_patch = H // self.n_split
        w_patch = W // self.n_split

        patches = x.view(B, C, self.n_split, h_patch, self.n_split, w_patch)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, self.n_split**2, C, h_patch, w_patch)
        
        num_patches = patches.shape[1]
        

        indices = torch.stack([torch.randperm(num_patches) for _ in range(B)]).to(x.device)
        
        shuffled_patches = torch.zeros_like(patches)
        for b in range(B):
            shuffled_patches[b] = patches[b, indices[b]]

        shuffled_patches = shuffled_patches.view(B, self.n_split, self.n_split, C, h_patch, w_patch)
        shuffled_patches = shuffled_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        mixed_img = shuffled_patches.view(B, C, H, W)
        
        return mixed_img, indices

    def restore_tokens(self, tokens, indices):
        """
        Restores ViT tokens [B, N, D] based on indices derived from Block Shuffle.
        """
        B, N, D = tokens.shape
        if indices is None:
            return tokens
        

        H_grid = W_grid = int(math.sqrt(N)) 

        patches_per_block_h = H_grid // self.n_split
        patches_per_block_w = W_grid // self.n_split
        
   
        x = tokens.view(B, H_grid, W_grid, D)
 
        x = x.view(B, self.n_split, patches_per_block_h, self.n_split, patches_per_block_w, D)
        x = x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, self.n_split**2, patches_per_block_h, patches_per_block_w, D)
        
        restore_indices = torch.argsort(indices, dim=1)
        restored_x = torch.zeros_like(x)
        for b in range(B):
            restored_x[b] = x[b, restore_indices[b]]

        restored_x = restored_x.view(B, self.n_split, self.n_split, patches_per_block_h, patches_per_block_w, D)
        restored_x = restored_x.permute(0, 1, 3, 2, 4, 5).contiguous().view(B, H_grid, W_grid, D)
        
        # Flatten
        return restored_x.view(B, N, D)

    def restore_spatial(self, feature_map, indices):
        """
        Restores a spatial feature map [B, C, H, W] (e.g. Skip Connections).
        """
        B, C, H, W = feature_map.shape
        if indices is None: return feature_map
        
        h_patch = H // self.n_split
        w_patch = W // self.n_split
        
        patches = feature_map.view(B, C, self.n_split, h_patch, self.n_split, w_patch)
        patches = patches.permute(0, 2, 4, 1, 3, 5).contiguous().view(B, self.n_split**2, C, h_patch, w_patch)
        
        restore_indices = torch.argsort(indices, dim=1)
        restored_patches = torch.zeros_like(patches)
        for b in range(B):
            restored_patches[b] = patches[b, restore_indices[b]]
            
        restored_patches = restored_patches.view(B, self.n_split, self.n_split, C, h_patch, w_patch)
        restored_patches = restored_patches.permute(0, 3, 1, 4, 2, 5).contiguous()
        return restored_patches.view(B, C, H, W)


class Attention(nn.Module):
    def __init__(self, config, vis):
        super(Attention, self).__init__()
        self.vis = vis
        self.num_attention_heads = config.transformer["num_heads"]
        self.attention_head_size = int(config.hidden_size / self.num_attention_heads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = Linear(config.hidden_size, self.all_head_size)
        self.key = Linear(config.hidden_size, self.all_head_size)
        self.value = Linear(config.hidden_size, self.all_head_size)

        self.out = Linear(config.hidden_size, config.hidden_size)
        self.attn_dropout = Dropout(config.transformer["attention_dropout_rate"])
        self.proj_dropout = Dropout(config.transformer["attention_dropout_rate"])

        self.softmax = Softmax(dim=-1)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(self, hidden_states):
        mixed_query_layer = self.query(hidden_states)
        mixed_key_layer = self.key(hidden_states)
        mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        attention_probs = self.softmax(attention_scores)
        weights = attention_probs if self.vis else None
        attention_probs = self.attn_dropout(attention_probs)

        context_layer = torch.matmul(attention_probs, value_layer)
        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)
        attention_output = self.out(context_layer)
        attention_output = self.proj_dropout(attention_output)
        return attention_output, weights


class Mlp(nn.Module):
    def __init__(self, config):
        super(Mlp, self).__init__()
        self.fc1 = Linear(config.hidden_size, config.transformer["mlp_dim"])
        self.fc2 = Linear(config.transformer["mlp_dim"], config.hidden_size)
        self.act_fn = ACT2FN["gelu"]
        self.dropout = Dropout(config.transformer["dropout_rate"])

        self._init_weights()

    def _init_weights(self):
        nn.init.xavier_uniform_(self.fc1.weight)
        nn.init.xavier_uniform_(self.fc2.weight)
        nn.init.normal_(self.fc1.bias, std=1e-6)
        nn.init.normal_(self.fc2.bias, std=1e-6)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act_fn(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


class Embeddings(nn.Module):
    """Construct the embeddings from patch, position embeddings.
    """

    def __init__(self, config, img_size, in_channels=3):
        super(Embeddings, self).__init__()
        self.hybrid = None
        self.config = config
        img_size = _pair(img_size)

        if config.patches.get("grid") is not None:  # ResNet
            grid_size = config.patches["grid"]
            patch_size = (img_size[0] // 16 // grid_size[0], img_size[1] // 16 // grid_size[1])
            patch_size_real = (patch_size[0] * 16, patch_size[1] * 16)
            n_patches = (img_size[0] // patch_size_real[0]) * (img_size[1] // patch_size_real[1])  
            self.hybrid = True
        else:
            patch_size = _pair(config.patches["size"])
            n_patches = (img_size[0] // patch_size[0]) * (img_size[1] // patch_size[1])
            self.hybrid = False

        if self.hybrid:
            self.hybrid_model = ResNetV2(block_units=config.resnet.num_layers, width_factor=config.resnet.width_factor)
            in_channels = self.hybrid_model.width * 16
        self.patch_embeddings = Conv2d(in_channels=in_channels,
                                       out_channels=config.hidden_size,
                                       kernel_size=patch_size,
                                       stride=patch_size)
        self.position_embeddings = nn.Parameter(torch.zeros(1, n_patches, config.hidden_size))

        self.dropout = Dropout(config.transformer["dropout_rate"])

    def forward(self, x):
        if self.hybrid:
            x, features = self.hybrid_model(x)
        else:
            features = None
        x = self.patch_embeddings(x)  
        x = x.flatten(2)
        x = x.transpose(-1, -2)  

        embeddings = x + self.position_embeddings
        embeddings = self.dropout(embeddings)
        return embeddings, features


class Block(nn.Module):
    def __init__(self, config, vis):
        super(Block, self).__init__()
        self.hidden_size = config.hidden_size
        self.attention_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn_norm = LayerNorm(config.hidden_size, eps=1e-6)
        self.ffn = Mlp(config)
        self.attn = Attention(config, vis)

    def forward(self, x):
        h = x
        x = self.attention_norm(x)
        x, weights = self.attn(x)
        x = x + h

        h = x
        x = self.ffn_norm(x)
        x = self.ffn(x)
        x = x + h
        return x, weights

    def load_from(self, weights, n_block):
        ROOT = f"Transformer/encoderblock_{n_block}"
        with torch.no_grad():
            query_weight = np2th(weights[pjoin(ROOT, ATTENTION_Q, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            key_weight = np2th(weights[pjoin(ROOT, ATTENTION_K, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            value_weight = np2th(weights[pjoin(ROOT, ATTENTION_V, "kernel")]).view(self.hidden_size, self.hidden_size).t()
            out_weight = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "kernel")]).view(self.hidden_size, self.hidden_size).t()

            query_bias = np2th(weights[pjoin(ROOT, ATTENTION_Q, "bias")]).view(-1)
            key_bias = np2th(weights[pjoin(ROOT, ATTENTION_K, "bias")]).view(-1)
            value_bias = np2th(weights[pjoin(ROOT, ATTENTION_V, "bias")]).view(-1)
            out_bias = np2th(weights[pjoin(ROOT, ATTENTION_OUT, "bias")]).view(-1)

            self.attn.query.weight.copy_(query_weight)
            self.attn.key.weight.copy_(key_weight)
            self.attn.value.weight.copy_(value_weight)
            self.attn.out.weight.copy_(out_weight)
            self.attn.query.bias.copy_(query_bias)
            self.attn.key.bias.copy_(key_bias)
            self.attn.value.bias.copy_(value_bias)
            self.attn.out.bias.copy_(out_bias)

            mlp_weight_0 = np2th(weights[pjoin(ROOT, FC_0, "kernel")]).t()
            mlp_weight_1 = np2th(weights[pjoin(ROOT, FC_1, "kernel")]).t()
            mlp_bias_0 = np2th(weights[pjoin(ROOT, FC_0, "bias")]).t()
            mlp_bias_1 = np2th(weights[pjoin(ROOT, FC_1, "bias")]).t()

            self.ffn.fc1.weight.copy_(mlp_weight_0)
            self.ffn.fc2.weight.copy_(mlp_weight_1)
            self.ffn.fc1.bias.copy_(mlp_bias_0)
            self.ffn.fc2.bias.copy_(mlp_bias_1)

            self.attention_norm.weight.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "scale")]))
            self.attention_norm.bias.copy_(np2th(weights[pjoin(ROOT, ATTENTION_NORM, "bias")]))
            self.ffn_norm.weight.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "scale")]))
            self.ffn_norm.bias.copy_(np2th(weights[pjoin(ROOT, MLP_NORM, "bias")]))


class BlockAdapter(nn.Module):
    def __init__(self, block: Block):
        super().__init__()
        self.block = block

    def forward(self, x, H=None, W=None):
        x, _ = self.block(x)
        return x


class Encoder(nn.Module):
    def __init__(self, config, vis):
        super(Encoder, self).__init__()
        self.vis = vis
        self.layer = nn.ModuleList()
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        for _ in range(config.transformer["num_layers"]):
            layer = Block(config, vis)
            self.layer.append(copy.deepcopy(layer))

    def forward(self, hidden_states):
        attn_weights = []
        for layer_block in self.layer:
            hidden_states, weights = layer_block(hidden_states)
            if self.vis:
                attn_weights.append(weights)
        encoded = self.encoder_norm(hidden_states)
        return encoded, attn_weights


class EdgeEnhancedEncoder(nn.Module):
    def __init__(self, config, vis, n_stages=3):
        super().__init__()
        self.vis = vis
        self.blocks = nn.ModuleList([BlockAdapter(Block(config, vis)) for _ in range(config.transformer["num_layers"])])
        self.encoder_norm = LayerNorm(config.hidden_size, eps=1e-6)
        L = len(self.blocks)
        base = L // n_stages
        rem = L % n_stages
        self.stage_slices = []
        s = 0
        for i in range(n_stages):
            e = s + base + (1 if i < rem else 0)
            self.stage_slices.append((s, e))
            s = e

        self.stages = nn.ModuleList([
            InteractionBlock(
                dim=config.hidden_size,
                num_heads=config.transformer["num_heads"],
                n_points=4, deform_ratio=1.0,
                with_cffn=True, cffn_ratio=0.25, drop=0.0, drop_path=0.0,
                extra_extractor=False, with_cp=False
            ) for _ in range(n_stages)
        ])
        

        self.fusion_modules = nn.ModuleList([
            nn.Sequential(
                nn.Linear(config.hidden_size * 2, config.hidden_size),
                nn.LayerNorm(config.hidden_size),
                nn.GELU()
            ) for _ in range(n_stages)
        ])

    def forward(self, x_tokens, c_tokens, deform_inputs1, deform_inputs2, Hc, Wc, Ht, Wt):
        prev_features = x_tokens.clone()
        
        for stage_idx, ((s, e), stage, fusion) in enumerate(zip(self.stage_slices, self.stages, self.fusion_modules)):
            blocks = self.blocks[s:e]
            
            injector_output, c_tokens = stage(
                x_tokens, c_tokens, blocks,
                deform_inputs1, deform_inputs2,
                Ht, Wt, Hc, Wc,  
                return_injector_output=True
            )
            fused_input = torch.cat([injector_output, prev_features], dim=-1)
            fused_output = fusion(fused_input)

            x_tokens = fused_output
            for block in blocks:
                x_tokens = block(x_tokens, Ht, Wt)

            c_tokens = stage.extractor(
                query=c_tokens,
                reference_points=deform_inputs2[0],
                feat=x_tokens,
                spatial_shapes=deform_inputs2[1],
                level_start_index=deform_inputs2[2],
                H=Hc,  
                W=Wc,  
            )
            prev_features = x_tokens.clone()
        
        x_tokens = self.encoder_norm(x_tokens)
        attn_weights = []
        return x_tokens, attn_weights


class Transformer(nn.Module):
    def __init__(self, config, img_size, vis, use_ssa_in_encoder=False):
        super(Transformer, self).__init__()
        self.embeddings = Embeddings(config, img_size=img_size)
        self.encoder = EdgeEnhancedEncoder(config, vis, n_stages=3)
        
        self.psc_handler = PairShuffleMod(n_split=4)

        self.use_ssa_in_encoder = use_ssa_in_encoder
        if self.use_ssa_in_encoder:
            resnet_deep_channel = self.embeddings.hybrid_model.width * 8 if self.embeddings.hybrid else config.hidden_size
            self.ssa_after_resnet = SSA_ConvBlock(resnet_deep_channel)

        self.edge_prior = EdgePriorModule(
            block=BasicBlock,
            num_blocks=[2, 2, 2, 2],
            embed_dim=config.hidden_size
        )

        if self.embeddings.hybrid:
            grid = self.embeddings.config.patches["grid"]
            ps = (img_size // 16 // grid[0]) * 16
            self.patch_size = ps
        else:
            self.patch_size = self.embeddings.config.patches["size"]

    def forward(self, input_ids, psc=False):
        """
        Modified forward to support internal Pair Shuffle Consistency.
        Args:
            input_ids: Input image tensor [B, C, H, W]
            psc (bool): Whether to enable Pair Shuffle Consistency for this pass
        """
        x_input = input_ids
        shuffle_indices = None
        
        if psc and self.training:
            x_input, shuffle_indices = self.psc_handler.shuffle(x_input)
        embedding_output, features = self.embeddings(x_input) 
        
        if self.use_ssa_in_encoder and features is not None:
            features[0] = self.ssa_after_resnet(features[0])

        B, Nt, C = embedding_output.shape
        Ht = Wt = int(np.sqrt(Nt))

        rgb = x_input if x_input.size(1) == 3 else x_input.repeat(1, 3, 1, 1)

        c1, c2, c3, c4 = self.edge_prior(rgb)
        c_tokens = torch.cat([c2, c3, c4], dim=1)
        deform_inputs1, deform_inputs2 = deform_inputs(rgb, self.patch_size)
        Hc, Wc = rgb.shape[-2] // 8, rgb.shape[-1] // 8
        encoded, attn_weights = self.encoder(embedding_output, c_tokens, deform_inputs1, deform_inputs2, Hc, Wc, Ht, Wt)
        
        if psc and self.training and shuffle_indices is not None:
            encoded = self.psc_handler.restore_tokens(encoded, shuffle_indices)
            if features is not None:
                restored_features = []
                for f in features:
                    if f is not None:
                        restored_f = self.psc_handler.restore_spatial(f, shuffle_indices)
                        restored_features.append(restored_f)
                    else:
                        restored_features.append(None)
                features = restored_features

        return encoded, attn_weights, features


class Conv2dReLU(nn.Sequential):
    def __init__(
            self,
            in_channels,
            out_channels,
            kernel_size,
            padding=0,
            stride=1,
            use_batchnorm=True,
    ):
        conv = nn.Conv2d(
            in_channels,
            out_channels,
            kernel_size,
            stride=stride,
            padding=padding,
            bias=not (use_batchnorm),
        )
        relu = nn.ReLU(inplace=True)

        bn = nn.BatchNorm2d(out_channels)

        super(Conv2dReLU, self).__init__(conv, bn, relu)



class DecoderBlock(nn.Module):
    def __init__(
            self,
            in_channels,
            out_channels,
            skip_channels=0,
            use_batchnorm=True,
    ):
        super().__init__()
        self.conv1 = Conv2dReLU(
            in_channels + skip_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.conv2 = Conv2dReLU(
            out_channels,
            out_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=use_batchnorm,
        )
        self.up = nn.UpsamplingBilinear2d(scale_factor=2)

    def forward(self, x, skip=None):
        x = self.up(x)
        if skip is not None:
            x = torch.cat([x, skip], dim=1)
        x = self.conv1(x)
        x = self.conv2(x)
        return x


class DecoderCup(nn.Module):
    def __init__(self, config):
        super().__init__()
        self.config = config
        head_channels = 512
        self.conv_more = Conv2dReLU(
            config.hidden_size,
            head_channels,
            kernel_size=3,
            padding=1,
            use_batchnorm=True,
        )
        decoder_channels = config.decoder_channels
        in_channels = [head_channels] + list(decoder_channels[:-1])
        out_channels = decoder_channels

        if self.config.n_skip != 0:
            skip_channels = self.config.skip_channels
            for i in range(4-self.config.n_skip): 
                skip_channels[3-i]=0
        else:
            skip_channels=[0,0,0,0]

        blocks = [
            DecoderBlock(in_ch, out_ch, sk_ch) for in_ch, out_ch, sk_ch in zip(in_channels, out_channels, skip_channels)
        ]
        self.blocks = nn.ModuleList(blocks)

    def forward(self, hidden_states, features=None):
        B, n_patch, hidden = hidden_states.size()
        h, w = int(np.sqrt(n_patch)), int(np.sqrt(n_patch))
        x = hidden_states.permute(0, 2, 1)
        x = x.contiguous().view(B, hidden, h, w)
        x = self.conv_more(x)
        for i, decoder_block in enumerate(self.blocks):
            if features is not None:
                skip = features[i] if (i < self.config.n_skip) else None
            else:
                skip = None
            x = decoder_block(x, skip=skip)
        return x


class SegmentationHead(nn.Sequential):
    def __init__(self, in_channels, out_channels, kernel_size=3, upsampling=1):
        conv2d = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, padding=kernel_size // 2)
        upsampling = nn.UpsamplingBilinear2d(scale_factor=upsampling) if upsampling > 1 else nn.Identity()
        super().__init__(conv2d, upsampling)


class VisionTransformer(nn.Module):
    def __init__(self, config, img_size=224, num_classes=21843, zero_head=False, vis=False, use_ssa_in_encoder=False):
        super(VisionTransformer, self).__init__()
        self.num_classes = num_classes
        self.zero_head = zero_head
        self.classifier = config.classifier
        self.transformer = Transformer(config, img_size, vis, use_ssa_in_encoder=use_ssa_in_encoder)
        self.decoder = DecoderCup(config)
        self.segmentation_head = SegmentationHead(
            in_channels=config['decoder_channels'][-1],
            out_channels=config['n_classes'],
            kernel_size=3,
        )
        self.config = config

    def forward(self, x, psc=False):
        """
        Added psc argument to control Pair Shuffle Consistency from the top level.
        """
        if x.size()[1] == 1:
            x = x.repeat(1,3,1,1)
            
        x, attn_weights, features = self.transformer(x, psc=psc) 
        
        x = self.decoder(x, features)
        logits = self.segmentation_head(x)
        return logits

    def load_from(self, weights):
        with torch.no_grad():

            res_weight = weights
            self.transformer.embeddings.patch_embeddings.weight.copy_(np2th(weights["embedding/kernel"], conv=True))
            self.transformer.embeddings.patch_embeddings.bias.copy_(np2th(weights["embedding/bias"]))
            posemb = np2th(weights["Transformer/posembed_input/pos_embedding"])

            posemb_new = self.transformer.embeddings.position_embeddings
            if posemb.size() == posemb_new.size():
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            elif posemb.size()[1]-1 == posemb_new.size()[1]:
                posemb = posemb[:, 1:]
                self.transformer.embeddings.position_embeddings.copy_(posemb)
            else:
                logger.info("load_pretrained: resized variant: %s to %s" % (posemb.size(), posemb_new.size()))
                ntok_new = posemb_new.size(1)
                if self.classifier == "seg":
                    _, posemb_grid = posemb[:, :1], posemb[0, 1:]
                gs_old = int(np.sqrt(len(posemb_grid)))
                gs_new = int(np.sqrt(ntok_new))
                print('load_pretrained: grid-size from %s to %s' % (gs_old, gs_new))
                posemb_grid = posemb_grid.reshape(gs_old, gs_old, -1)
                zoom = (gs_new / gs_old, gs_new / gs_old, 1)
                posemb_grid = ndimage.zoom(posemb_grid, zoom, order=1)  
                posemb_grid = posemb_grid.reshape(1, gs_new * gs_new, -1)
                posemb = posemb_grid
                self.transformer.embeddings.position_embeddings.copy_(np2th(posemb))

            if self.transformer.embeddings.hybrid:
                self.transformer.embeddings.hybrid_model.root.conv.weight.copy_(np2th(res_weight["conv_root/kernel"], conv=True))
                gn_weight = np2th(res_weight["gn_root/scale"]).view(-1)
                gn_bias = np2th(res_weight["gn_root/bias"]).view(-1)
                self.transformer.embeddings.hybrid_model.root.gn.weight.copy_(gn_weight)
                self.transformer.embeddings.hybrid_model.root.gn.bias.copy_(gn_bias)

                for bname, block in self.transformer.embeddings.hybrid_model.body.named_children():
                    for uname, unit in block.named_children():
                        unit.load_from(res_weight, n_block=bname, n_unit=uname)


CONFIGS = {
    'ViT-B_16': configs.get_b16_config(),
    'ViT-B_32': configs.get_b32_config(),
    'ViT-L_16': configs.get_l16_config(),
    'ViT-L_32': configs.get_l32_config(),
    'ViT-H_14': configs.get_h14_config(),
    'R50-ViT-B_16': configs.get_r50_b16_config(),
    'R50-ViT-L_16': configs.get_r50_l16_config(),
    'testing': configs.get_testing(),
}

class SSA_ConvBlock(nn.Module):
    """
    Spatial Self-Attention (SSA) ConvBlock
    """
    def __init__(self, c):
        super().__init__()
        self.c = c
        self.fq = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.fk = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.fv = nn.Conv2d(c, c, kernel_size=3, padding=1, bias=False)
        self.bn = nn.BatchNorm2d(c)
        self.relu = nn.ReLU()

    def forward(self, inputs):
        B, C, H, W = inputs.shape
        HW = H * W
        if HW == 0: return inputs

        fq = self.fq(inputs).view(B, C, HW).permute(0, 2, 1)  
        fk = self.fk(inputs).view(B, C, HW)  
        fv = self.fv(inputs).view(B, C, HW).permute(0, 2, 1)  

        f_sim_tensor = torch.matmul(fq, fk) / (C ** 0.5)  
        scores = torch.softmax(f_sim_tensor, dim=-1)

        r = torch.matmul(scores, fv) 
        r = r.permute(0, 2, 1).view(B, C, H, W) 

        r = r + inputs
        r = self.bn(r)
        r = self.relu(r)
        return r