import time
import torch
import torch.optim as optim
import torch.backends.cudnn as cudnn
import torch.nn.functional as F
import torchvision.models as models
import numpy as np
from tqdm import tqdm
from torchvision import models
from torch.autograd import Variable
from PIL import Image
from torch import nn
from nets.TransUnet import get_transNet
from nets.unet_training import CE_Loss, Dice_loss
from utils.metrics import f_score
from torch.utils.data import DataLoader
from dataloader import unetDataset, unet_dataset_collate
from torch.optim import Optimizer
import math
import random  

class StochGradAdam(Optimizer):
    def __init__(self, params, lr=1e-3, betas=(0.9, 0.999), eps=1e-8, 
                 sampling_rate=0.8, weight_decay=0, beta1_decay=1.0):
        if not 0.0 <= lr: raise ValueError(f"无效的学习率: {lr}")
        if not 0.0 <= eps: raise ValueError(f"无效的epsilon值: {eps}")
        if not 0.0 <= betas[0] < 1.0: raise ValueError(f"无效的beta参数 (索引0): {betas[0]}")
        if not 0.0 <= betas[1] < 1.0: raise ValueError(f"无效的beta参数 (索引1): {betas[1]}")
        if not 0.0 < sampling_rate <= 1.0: raise ValueError(f"无效的采样率: {sampling_rate}")
        
        defaults = dict(lr=lr, betas=betas, eps=eps, 
                       sampling_rate=sampling_rate, 
                       weight_decay=weight_decay,
                       beta1_decay=beta1_decay)
        super(StochGradAdam, self).__init__(params, defaults)
    
    def __setstate__(self, state):
        super(StochGradAdam, self).__setstate__(state)
    
    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        
        for group in self.param_groups:
            beta1, beta2 = group['betas']
            lr = group['lr']
            eps = group['eps']
            sampling_rate = group['sampling_rate']
            weight_decay = group['weight_decay']
            beta1_decay = group['beta1_decay']
            
            for p in group['params']:
                if p.grad is None: continue
                grad = p.grad
                if weight_decay != 0: grad = grad.add(p, alpha=weight_decay)
                
                state = self.state[p]
                if len(state) == 0:
                    state['step'] = 0
                    state['exp_avg'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                    state['exp_avg_sq'] = torch.zeros_like(p, memory_format=torch.preserve_format)
                
                exp_avg, exp_avg_sq = state['exp_avg'], state['exp_avg_sq']
                state['step'] += 1
                t = state['step']
                
                mask = torch.rand_like(grad)
                omega = (mask < sampling_rate).float()
                sampled_grad = omega * grad
                
                beta1_t = beta1 * (beta1_decay ** (t - 1))
                exp_avg.mul_(beta1_t).add_(sampled_grad, alpha=(1 - beta1_t))
                exp_avg_sq.mul_(beta2).addcmul_(sampled_grad, sampled_grad, value=(1 - beta2))
                
                bias_correction1 = 1 - beta1 ** (t + 1)
                bias_correction2 = 1 - beta2 ** (t + 1)
                
                m_corr = exp_avg / bias_correction1
                v_corr = exp_avg_sq / bias_correction2
                
                denom = v_corr.sqrt().add_(eps)
                p.addcdiv_(m_corr, denom, value=-lr)
        return loss

def get_lr(optimizer):
    for param_group in optimizer.param_groups:
        return param_group['lr']

def fit_one_epoch(net, epoch, epoch_size, epoch_size_val, gen, genval, Epoch, cuda, aux_branch):
    net = net.train()


    total_loss = 0
    total_f_score = 0
    val_toal_loss = 0
    val_total_f_score = 0
    
    start_time = time.time()
    
    psc_probability = 0.8

    with tqdm(total=epoch_size, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(gen):
            if iteration >= epoch_size:
                break
            imgs, pngs, labels = batch

            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()

            optimizer.zero_grad()

            use_psc = True if random.random() < psc_probability else False

            if aux_branch:
    
                aux_outputs, outputs = net(imgs, psc=use_psc)
                aux_loss = CE_Loss(aux_outputs, pngs, num_classes=NUM_CLASSES)
                main_loss = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
                loss = aux_loss + main_loss
                if dice_loss:
                    aux_dice = Dice_loss(aux_outputs, labels)
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + aux_dice + main_dice

            else:
                outputs = net(imgs, psc=use_psc)
                
                loss = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
                if dice_loss:
                    main_dice = Dice_loss(outputs, labels)
                    loss = loss + main_dice

            with torch.no_grad():
                _f_score = f_score(outputs, labels)

            loss.backward()
            optimizer.step()

            total_loss += loss.item()
            total_f_score += _f_score.item()

            waste_time = time.time() - start_time
            pbar.set_postfix(**{'total_loss': total_loss / (iteration + 1),
                                'f_score': total_f_score / (iteration + 1),
                                's/step': waste_time,
                                'lr': get_lr(optimizer)})
            pbar.update(1)
            start_time = time.time()

    print('Start Validation')
    with tqdm(total=epoch_size_val, desc=f'Epoch {epoch + 1}/{Epoch}', postfix=dict, mininterval=0.3) as pbar:
        for iteration, batch in enumerate(genval):
            if iteration >= epoch_size_val:
                break
            imgs, pngs, labels = batch
            with torch.no_grad():
                imgs = Variable(torch.from_numpy(imgs).type(torch.FloatTensor))
                pngs = Variable(torch.from_numpy(pngs).type(torch.FloatTensor)).long()
                labels = Variable(torch.from_numpy(labels).type(torch.FloatTensor))
                if cuda:
                    imgs = imgs.cuda()
                    pngs = pngs.cuda()
                    labels = labels.cuda()
                if aux_branch:
                    aux_outputs, outputs = net(imgs)
                    aux_loss = CE_Loss(aux_outputs, pngs, num_classes=NUM_CLASSES)
                    main_loss = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
                    val_loss = aux_loss + main_loss
                    if dice_loss:
                        aux_dice = Dice_loss(aux_outputs, labels)
                        main_dice = Dice_loss(outputs, labels)
                        val_loss = val_loss + aux_dice + main_dice

                else:
                    outputs = net(imgs)
                    val_loss = CE_Loss(outputs, pngs, num_classes=NUM_CLASSES)
                    if dice_loss:
                        main_dice = Dice_loss(outputs, labels)
                        val_loss = val_loss + main_dice

                _f_score = f_score(outputs, labels)
                val_toal_loss += val_loss.item()
                val_total_f_score += _f_score.item()

            pbar.set_postfix(**{'total_loss': val_toal_loss / (iteration + 1),
                                'f_score': val_total_f_score / (iteration + 1),
                                'lr': get_lr(optimizer)})
            pbar.update(1)

    print('Finish Validation')
    print('Epoch:' + str(epoch + 1) + '/' + str(Epoch))
    print('Total Loss: %.4f || Val Loss: %.4f ' % (total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))

    totalBig_loss = ('%.4f' % (total_loss / (epoch_size + 1)))
    val_loss1232 = ('%.4f' % (val_toal_loss / (epoch_size_val + 1)))
    
    try:
        with open('train_loss.csv', mode='a+') as f:
            f.write(totalBig_loss + ',' + val_loss1232 + '\n')
        
        score = ('%.4f' % (val_total_f_score / (iteration + 1)))
        with open('acc.csv', mode='a+') as f:
            f.write(score + '\n')
    except Exception as e:
        print(f"Log writing error: {e}")

    print('Saving state, iter:', str(epoch + 1))
    torch.save(model.state_dict(), './logs/Epoch%d-Total_Loss%.4f-Val_Loss%.4f.pth' % (
    (epoch + 1), total_loss / (epoch_size + 1), val_toal_loss / (epoch_size_val + 1)))


if __name__ == "__main__":
    inputs_size = [256, 256, 3]
    log_dir = "./logs/"
    NUM_CLASSES = 2
    dice_loss = True
    pretrained = False
    backbone = "ECAresnet"
    aux_branch = False
    downsample_factor = 16
    Cuda = True

    model = get_transNet(n_classes=NUM_CLASSES).train()

    if Cuda:
        net = torch.nn.DataParallel(model)
        cudnn.benchmark = True
        net = net.cuda()

    with open(r"./Data/ImageSets/SegmentationClass/train.txt", "r") as f:
        train_lines = f.readlines()
    with open(r"./Data/ImageSets/SegmentationClass/val.txt", "r") as f:
        val_lines = f.readlines()

    if True:
        lr = 9e-4   
        Init_Epoch = 0
        Interval_Epoch = 100
        Batch_size = 4
        
        optimizer = StochGradAdam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            sampling_rate=0.8,
            weight_decay=1e-4,
            beta1_decay=1.0
        )
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=1, gamma=0.98)

        train_dataset = unetDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = unetDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=1, pin_memory=True,
                         drop_last=True, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=1, pin_memory=True,
                             drop_last=True, collate_fn=unet_dataset_collate)

        epoch_size = max(1, len(train_lines) // Batch_size)
        epoch_size_val = max(1, len(val_lines) // Batch_size)

        for epoch in range(Init_Epoch, Interval_Epoch):
            fit_one_epoch(model, epoch, epoch_size, epoch_size_val, gen, gen_val, Interval_Epoch, Cuda, aux_branch)
            lr_scheduler.step()

    if True:
        lr = 1e-5
        Interval_Epoch = 100
        Epoch = 200
        Batch_size = 4
        
        optimizer = StochGradAdam(
            model.parameters(),
            lr=lr,
            betas=(0.9, 0.999),
            eps=1e-8,
            sampling_rate=0.8,
            weight_decay=1e-4,
            beta1_decay=1.0
        )
        lr_scheduler = optim.lr_scheduler.StepLR(optimizer,step_size=1,gamma=0.98)

        train_dataset = unetDataset(train_lines, inputs_size, NUM_CLASSES, True)
        val_dataset = unetDataset(val_lines, inputs_size, NUM_CLASSES, False)
        gen = DataLoader(train_dataset, batch_size=Batch_size, num_workers=2, pin_memory=True,
                                drop_last=True, collate_fn=unet_dataset_collate)
        gen_val = DataLoader(val_dataset, batch_size=Batch_size, num_workers=2,pin_memory=True,
                                drop_last=True, collate_fn=unet_dataset_collate)

        epoch_size = max(1, len(train_lines)//Batch_size)
        epoch_size_val = max(1, len(val_lines)//Batch_size)

        for epoch in range(Interval_Epoch,Epoch):
            fit_one_epoch(model,epoch,epoch_size,epoch_size_val,gen,gen_val,Epoch,Cuda,aux_branch)
            lr_scheduler.step()