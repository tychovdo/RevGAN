## Memory-efficient Image-to-Image Translation in 2D and 3D
### [Tycho F.A. van der Ouderaa](https://tychovdo.github.io), [Daniel E. Worrall](https://deworrall92.github.io/)

## In CVPR 2019

We extend the Pix2pix and CycleGAN framework by exploring approximately invertible architectures in 2D and 3D. These architectures are approximately invertible by design and thus partially satisfy cycle-consistency before training even begins. Furthermore, since invertible architectures have constant memory complexity in depth, these models can be built arbitrarily deep without requiring additional memory. In the paper we demonstrate superior quantitative output on the Cityscapes and Maps datasets at near constant memory budget.

### [Paper](https://arxiv.org/abs/1902.02729) | [Code](https://github.com/tychovdo/RevGAN)



## Acknowledgements

We grateful to the Diagnostic Image Analysis Group (DIAG) of the Radboud University Medical Center, and in particular Prof. Dr. Bram van Ginneken for his collaboration on this project. We also thank the Netherlands Organisation for Scientific Research (NWO) for supporting this research and providing computational resources.

This  code relies heavily on the  image-to-image  translation  framework from [pytorch-CycleGAN-and-pix2pix](https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix) and extends the framework with paired and unpaired reversible models in 2D and 3D. The reversible blocks are implemented using a modified version of [MemCNN](https://github.com/silvandeleemput/memcnn).
