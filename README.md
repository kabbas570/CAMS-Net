# CAMS: Convolution and Attention-Free Mamba-based Cardiac Image Segmentation

## [Accepted at WACV 2025 (IEEE/CVF Winter Conference on Applications of Computer Vision)](https://wacv2025.thecvf.com/)


# Overview:
This paper demonstrates that convolution and self-attention, while widely used, are not the only effective methods for segmentation. Breaking with convention, we present a Convolution and self-attention-free Mamba-based semantic Segmentation Network named CAMS-Net for the task of medical image segmentation.

![image](https://raw.githubusercontent.com/kabbas570/CompSeg-MetaData/09d175b70f1e6c1a4b33c172531754a7eb72f4f1/figures/arch.png)

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



# Citation
@article{khan2024cams,

  title={CAMS: Convolution and Attention-Free Mamba-based Cardiac Image Segmentation},
  
  author={Khan, Abbas and Asad, Muhammad and Benning, Martin and Roney, Caroline and Slabaugh, Gregory},
  
  booktitle={Proceedings of the IEEE/CVF Winter Conference on Applications of Computer Vision},
  
  year={2025}
}

