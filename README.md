# ECAWU-Net
ECAWU-Net: A Structure-Preserving and Boundary-Aware CNN--Transformer Network for Retinal Vessel and Optic Disc Segmentation
ECAWU-Net is designed for fine-grained retinal vessel and optic disc segmentation in color fundus images. The network combines a hybrid CNN-Transformer backbone with multidimensional attention, edge-enhanced token interaction, and reversible random block shuffling.

## Highlights

- **SSA-CAW**: a multidimensional attention module that enhances spatial, cross-channel, and within-channel feature interactions.
- **EET**: an Edge-Enhanced Transformer encoder that introduces bidirectional interaction between semantic tokens and edge-sensitive features.
- **RST**: a Random Shuffle Tactic that performs reversible block-wise spatial perturbation during training and restores token/skip-feature alignment before decoding.
- **Edge supervision**: an auxiliary edge prediction branch is used to improve boundary localization.



## Installation

Create a clean Python environment:

```bash
conda create -n ecawu-net python=3.10 -y
conda activate ecawu-net
```

Install PyTorch according to your CUDA version. For example:

```bash
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118
```

Then install the remaining dependencies:

```bash
pip install -r requirements.txt
```

## Datasets

The experiments in the paper use two public datasets:

- **FIVES** for retinal vessel segmentation.
- **REFUGE2** for optic disc segmentation.


Each split file should contain one sample identifier per line, without the image suffix. The image and mask filenames should share the same identifier.


## Citation

If this code is useful for your research, please cite:

```bibtex
@article{ecawunet2026,
  title   = {ECAWU-Net: A Structure-Preserving and Boundary-Aware CNN-Transformer Network for Retinal Vessel and Optic Disc Segmentation},
  author  = {Gai, Rongli and Li, Longyu and Duan, Xiaoming},
  journal = {Computerized Medical Imaging and Graphics},
  year    = {2026}
}
```
