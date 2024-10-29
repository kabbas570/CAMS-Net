# CAMS: Convolution and Attention-Free Mamba-based Cardiac Image Segmentation

## [Accepted at WACV 2025 (IEEE/CVF Winter Conference on Applications of Computer Vision)](https://wacv2025.thecvf.com/)


# Overview:
This paper demonstrates that convolution and self-attention, while widely used, are not the only effective methods for segmentation. Breaking with convention, we present a Convolution and self-attention-free Mamba-based semantic Segmentation Network named CAMS-Net for the task of medical image segmentation.

![image](https://github.com/kabbas570/CAMS-Net/blob/052ac53b678a907be29aaca6b4abdd7dbd973d7a/figures/main_r2.png)

## Key Contributions:
***First Convolution and Self-attention-Free Architecture:*** To the best of our knowledge, we are the first to propose a convolution and self-attention-free Mamba-based segmentation network. 

***Linearly Interconnected Factorized Mamba (LIFM):***  LIFM block to reduce the trainable parameters of Mamba and improve its non-linearity. LIFM implements a weight-sharing strategy for different scanning directions, specifically for the two scanning direction strategies of vision Mamba, to reduce the computational complexity further whilst maintaining accuracy.

***Mamba Channel Mamba Spatial Aggregators:***   These modules learn information along the channel and spatial dimensions of the features, respectively.

## Evaluation:
Our approach was evaluated on two modalities using publicly available datasets:

***M&Ms-2 Dataset***

***CMR×Recon Segmentation Dataset***

Results demonstrate state-of-the-art segmentation performance across diverse cardiac imaging modalities.

# Training Steps

## Segmentation Model 

## ImageNet Pretrained Weights

[Mamba-Encoder for input size with spatial dim of 256 x 256](https://drive.google.com/open?id=1IMmrYufVxRek3sVfrY1FZDax0NVpQy7g&usp=drive_copy)

[Mamba-Encoder for input size with spatial dim of 160 x 160](https://drive.google.com/open?id=1zmxagS6x7_osxNpoQxICSvNltVuNxJTC&usp=drive_copy)





# Citation
@article{khan2024cams,

  title={CAMS: Convolution and Attention-Free Mamba-based Cardiac Image Segmentation},
  
  author={Khan, Abbas and Asad, Muhammad and Benning, Martin and Roney, Caroline and Slabaugh, Gregory},
  
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  
  year={2025}
}

