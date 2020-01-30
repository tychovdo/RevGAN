## RevGAN: Memory-efficient Image-to-Image Translation in 2D and 3D

RevGAN implementation in PyTorch. We extend the Pix2pix and CycleGAN framework by exploring approximately invertible architectures in 2D and 3D. These architectures are approximately invertible by design and thus partially satisfy cycle-consistency before training even begins. Furthermore, since invertible architectures have constant memory complexity in depth, these models can be built arbitrarily deep without requiring additional memory. In the paper we demonstrate superior quantitative output on the Cityscapes and Maps datasets at near constant memory budget.

### [Project](https://github.com/tychovdo/RevGAN) | [Paper](https://arxiv.org/abs/1902.02729) <br>
<br>
Reversible GANs for Memory-efficient Image-to-Image Translation <br>
Tycho F.A. van der Ouderaa, Daniel E. Worral <br>
University of Amsterdam, CVPR 2019 <br>
<br>

```
@inproceedings{vanderOuderaa2019revgan,
title={Reversible GANs for Memory-efficient Image-to-Image Translation},
author={van der Ouderaa, Worrall, Daniel E},
booktitle={CVPR 2019},
}
```

## Getting Started
### Installation
- Install [PyTorch](http://pytorch.org and) 0.4+ and other dependencies (e.g., torchvision, [visdom](https://github.com/facebookresearch/visdom) and [dominate](https://github.com/Knio/dominate)).
  - For pip users, please type the command `pip install -r requirements.txt`.
- Clone this repo:
```bash
git clone https://github.com/tychovdo/RevGAN
cd RevGAN
```

### Training

- Download a CycleGAN dataset (e.g. cityscapes):
```
bash ./datasets/download_cyclegan_dataset.sh cityscapes
```
- Train a model:
```
python train.py --dataroot ./datasets/cityscapes --name example_run --model unpaired_revgan
```

During training, results will be saved to `./checkpoints/example_run/web/index.html`

### Testing

- Test a model:
```
python test.py --dataroot ./datasets/cityscapes --name example_run --model unpaired_revgan
```

The test results will be saved to `./results/example_run/test_latest/index.html`


## Prerequisites
- Linux or macOS
- Python 2 or 3
- NVIDIA GPU + CUDA cuDNN


## Acknowledgements

We grateful to the Diagnostic Image Analysis Group (DIAG) of the Radboud University Medical Center, and in particular Prof. Dr. Bram van Ginneken for his collaboration on this project. We also thank the Netherlands Organisation for Scientific Research (NWO) for supporting this research and providing computational resources.

This  code relies heavily on the  image-to-image  translation  framework from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and extends the framework with paired and unpaired reversible models in 2D and 3D. The reversible blocks are implemented using a modified version of [MemCNN](https://github.com/silvandeleemput/memcnn).
