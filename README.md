<div align="center">
  
## Denoising-Enhanced Coarse-to-Fine Infrared Small Target Detection with Attention Prior-Guided Knowledge Distillation

Authors: Houzhang Fang<sup>1</sup>, Ruixuan Huang<sup>1</sup>, Qiuhuan Chen<sup>1</sup>, Xiaolin Wang<sup>1</sup>, Yi Chang<sup>2</sup>, Luxin Yan<sup>2</sup>
  
<sup>1</sup>Xidian University,  <sup>2</sup>Huazhong University of Science and Technology

**European Conference on Computer Vision (ECCV) 2026**

<h4>
  <a href="https://arxiv.org/abs/2606.21956">[Paper PDF|arXiv]</a>
  <a href="#citation">[BibTex]</a>
</h4>

</div>

## Abstract

Infrared small target detection (IRSTD) in high-resolution images is crucial for many practical applications, such as surveillance of unmanned aerial vehicles (UAVs) and UAV-based ground monitoring. However, IRSTD remains challenging due to the small size and weak features of targets, as well as significant interference from complex dynamic backgrounds. Existing detection methods often suffer from redundant computations on non-target background regions and insufficient exploitation of target context information, which limits their performance in complex backgrounds. To address these issues, we propose an efficient coarse-to-fine infrared small target detection framework with attention prior-guided knowledge distillation, termed ECFNet. In the coarse stage, we design a region binary classification network (RBCN) on grid-based multi-scale feature maps to efficiently recognize target-containing context region proposals. Moreover, we introduce a novel denoising-assisted training strategy that incorporates noisy ground-truth (GT) masks into RBCN feature maps. By training the network to reconstruct the GT masks via a denoising task, this strategy encourages the model to explicitly capture target-background context, thereby enhancing its ability to distinguish target proposals from background regions.  In the fine stage, we customize a lightweight target detector to the coarse stage’s region proposals for balancing accuracy and efficiency. Furthermore, we propose a knowledge distillation strategy guided by the teacher-student cross-attention prior. This mechanism directs the student to focus on critical target regions, thereby enhancing the discriminative feature representation for infrared small targets. Extensive experiments on three real infrared datasets demonstrate that our method outperforms both existing single-stage and two-stage approaches while maintaining high real-time processing efficiency.

## Network Architecture
<p align="center">
  <img src="coarse_to_fine_framework.png" width="auto" alt="accessibility text">
</p>
Overall pipeline of the proposed ECFNet.

## Dataset Download Link

Our custom-built Car dataset comprises 18,230 infrared images, each with a spatial resolution of $640 \times 512$ pixels. Car dataset can be downloaded via the following link:

- Download Dataset Here -[Baidu Netdisk](<https://pan.baidu.com/s/1XUSNjVa8vikR6TQOtMwkTQ?pwd=xdut>),  [Google Drive(TODO)](<>).
  
If you find the Car dataset useful for your research, please consider citing our paper  [[`BibTex`](#citation)]. Your acknowledgement is greatly appreciated!

## Citation
If you find our work and dataset useful for your research, please consider citing our paper. Thank you!
```bibtex
@inproceedings{2026ECCV_ECFNet,
    title     = {Denoising-Enhanced Coarse-to-Fine Infrared Small Target Detection with Attention Prior-Guided Knowledge Distillation},
    author    = {Houzhang Fang and Ruixuan Huang and Qiuhuan Chen and Xiaolin Wang and Yi Chang and Luxin Yan},
    booktitle = {European Conference on Computer Vision (ECCV)},
    year      = {2026},
    pages     = { },
}
```

## Contact
If you have any question, please contact: houzhangfang@xidian.edu.cn and hrx@stu.xidian.edu.cn,

Copyright &copy; Xidian University.

## License

This dataset is licensed under the Creative Commons
Attribution-NonCommercial-NoDerivatives 4.0 International License
(CC BY-NC-ND 4.0):
https://creativecommons.org/licenses/by-nc-nd/4.0/

The dataset may be used for non-commercial academic research and
educational purposes without prior permission, provided that the dataset
and the associated paper are properly cited.

Commercial use and the public release or redistribution of any modified,
derived, augmented, re-annotated, or repackaged dataset require prior
written permission from the authors.
