<h1 align="center">
  <br>
  <a href="#">
    <picture>
      <source media="(prefers-color-scheme: dark)" srcset="logo-retinex-dark.svg">
      <source media="(prefers-color-scheme: light)" srcset="logo-retinex-light.svg">
      <img src="logo-retinex-light.svg" alt="Retinex-Based Low-Light Image Enhancement" width="420">
    </picture>
  </a>
  <br>
</h1>




This is a final project for the Computer Vision course in [MUST](https://www.must.edu.mo), which focuses on low-light image enhancement based on Retinex theory. 
It reproduces and compares three representative methods: Multi-Scale Retinex (MSR), Retinex-Net, and Retinexformer, under a unified experimental and evaluation pipeline.

**Authors:**  
- Shuoyan Tang (12200009721)  
- Xinran Mi (12200011281)

### Introduction

Low-light image enhancement is crucial for applications in photography, autonomous driving,and surveillance. Images captured under insufficient illumination often suffer from noise, low
contrast, and color distortion, making visibility improvement in low-light conditions a critical topic. This project aims to reproduce and analyze three representative Retinex-based enhancement approaches: the classical Retinex theory, the CNN-based Retinex-Net, the Transformer-based Retinexformer. We implemented classical Retinex algorithms in Python, and re-training Retinex-Net and Retinexformer following the authors’ official settings. Comparative experiments will be conducted on public benchmark ( LOL-v1). We report PSNR and SSIM for quantitative
evaluation of enhancement quality and complement them with analyses of illumination estimation and detail recovery.


### Main Results (PSNR and SSIM)
|     Category     |    Method        |    PSNR   | SSIM  |
| ---------------- |:----------------:|:---------:|:-----:|
|Classical	       |Retinex(Gray)     |11.9213    |0.4770 |
|Classical	       |Retinex(RGB)      |13.3530    |0.4749 |
|Deep	           |Retinex-Net       |16.7740    |0.4285 |
|Deep	           |Retinexformer     |13.6988    |0.7732 |


Table 1: Qualitative results of the deep Retinex-based methods on LOL-v1



### Requirements: software

> Below is the software stack we actually used on the Alibaba Cloud GPU server to run Multi-scale Retinex (IPOL C code), Retinex-Net, and Retinexformer.

(0)  Server Environment
- _Cloud Platform_: Alibaba Cloud ECS  
- _CPU Architecture_: x86_64  
- _Operating System_: Ubuntu 20.04.6 LTS

(1)  Python & Conda  Environment
- _Miniconda3_
- _Conda_: conda 25.9.1

(2) Multiscale Retinex Environment (IPOL C implementation, no conda env)
- _Compiler & build toolchain_: gcc / g++ , make
- _Image / math libraries_: libpng, libjpeg, FFTW3
-_ Command-line utilities_: bash, wget/curl, tar
- _Optional Python stack_ : numpy, Pillow, opencv-python, scikit-image, pandas

(3) Retinex-Net Environment (conda env: Retinexnet)
- _Python_: Python 3.7
- _Core DL stack_: tensorflow 1.x (CPU build),keras 2.x
- _Common Python packages_: numpy,scipy,scikit-image,opencv-python (cv2),Pillow,matplotlib
- _Training / data utilities_: h5py,tqdm
- _Config / logging tools_: pyyaml,tensorboard

(4)  Retinexformer Environment (conda env: Retinexformer)
- _Python_: Python 3.7
- _Core DL / Restoration stack_: torch,basicsr
- _Data / IO backend_: python-lmdb
- _Common Python packages_: numpy,opencv-python (cv2)
- _Config / utilities / bulid toolchain_: pyyaml,tqdm,Cython,cffi,setuptools


### Pretrained Models

All pretrained models are already packed inside their own subfolders — just run the code and everything works out of the box.



### Preparation for Training
1.	You need to download the LOL-v1 dataset using  [Baidu Disk](https://pan.baidu.com/s/1ZAC9TWR-YeuLIkWs3L7z4g?pwd=cyh2) (code: `cyh2`) or [Google Drive](https://drive.google.com/file/d/1L-kqSQyrmMueBh_ziWoPFhfsAh50h20H/view?usp=sharing)

2.	following the readme from the subfolder of each method ( [Multi-retienx](Multiscale-retinex/readme(Multi-retinex).md) ; [Retinex-net](Retinex-net/readme(retinex-net).md) ; [Retinexformer](Retinexformer/readme(retinexformer).md) to run each of them.

3.	After getting the enhance image from above three method, we can using the [evaluation code](Evaluation/PSNR-SSIM.py)  to know the each method's PSNR and SSIM and make a diretly comparison.








