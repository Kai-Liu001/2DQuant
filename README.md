# 2DQuant: Low-bit Post-Training Quantization for Image Super-Resolution

[Kai Liu](https://kai-liu001.github.io/), [Haotong Qin](https://htqin.github.io/), [Yong Guo](https://www.guoyongcs.com/), [Xin Yuan](https://en.westlake.edu.cn/faculty/xin-yuan.html), [Linghe Kong](https://www.cs.sjtu.edu.cn/~linghe.kong/), [Guihai Chen](https://cs.nju.edu.cn/gchen/index.htm), and [Yulun Zhang](http://yulunzhang.com/), "2DQuant: Low-bit Post-Training Quantization for Image Super-Resolution", arXiv, 2024

[[arXiv]()] [visual results] [pretrained models]



#### 🔥🔥🔥 News

- **2024-06-09:** This repo is released.

---

> **Abstract:** Low-bit quantization has become widespread for compressing image super-resolution (SR) models for edge deployment, which allows advanced SR models to enjoy compact low-bit parameters and efficient integer/bitwise constructions for storage compression and inference acceleration, respectively. However, it is notorious that low-bit quantization degrades the accuracy of SR models compared to their full-precision (FP) counterparts. Despite several efforts to alleviate the degradation, the transformer-based SR model still suffers severe degradation due to its distinctive activation distribution. In this work, we present a dual-stage low-bit post-training quantization (PTQ) method for image super-resolution, namely 2DQuant, which achieves efficient and accurate SR under low-bit quantization. The proposed method first investigates the weight and activation and finds that the distribution is characterized by coexisting symmetry and asymmetry, long tails. Specifically, we propose Distribution-Oriented Bound Initialization (DOBI), using different searching strategies to search a coarse bound for quantizers. To obtain refined quantizer parameters, we further propose Distillation Quantization Calibration (DQC), which employs a distillation approach to make the quantized model learn from its FP counterpart. Through extensive experiments on different bits and scaling factors, the performance of DOBI can reach the state-of-the-art (SOTA) while after stage two, our method surpasses existing PTQ in both metrics and visual effects. 2DQuant gains an increase in PSNR as high as 4.52dB on Set5 ($\times 2$) compared with SOTA when quantized to 2-bit and enjoys a 3.60 $\times$ compression ratio and 5.08 $\times$ speedup ratio.

![](figures/pipeline.png)


---

---

|                            HR                             |                               LR                               | [SwinIR-light (FP)](https://github.com/JingyunLiang/SwinIR) |          [DBDC+Pac](https://openaccess.thecvf.com/content/CVPR2023/html/Tu_Toward_Accurate_Post-Training_Quantization_for_Image_Super_Resolution_CVPR_2023_paper.html)          |                         2DQuant (ours)                         |
|:---------------------------------------------------------:|:--------------------------------------------------------------:|:---------------------------------------------------------------------------------------:|:-----------------------------------------------------------:|:----------------------------------------------------------------:|
| <img src="figures/comp/img072-gt.png" height=80> | <img src="figures/comp/img072-bicubic.png" height=80> |               <img src="figures/comp/img072-fp.png" height=80>                | <img src="figures/comp/img072-pac.png" height=80> | <img src="figures/comp/img072-ours.png" height=80> |
| <img src="figures/comp/img092-gt.png" height=80> | <img src="figures/comp/img092-bicubic.png" height=80> |               <img src="figures/comp/img092-fp.png" height=80>                | <img src="figures/comp/img092-pac.png" height=80> | <img src="figures/comp/img092-ours.png" height=80> |


## TODO

* [ ] Release code and pretrained models

## Contents

1. Datasets
1. Models
1. Training
1. Testing
1. [Results](#results)
1. [Citation](#citation)
1. [Acknowledgements](#acknowledgements)

## <a name="results"></a> Results

We achieved state-of-the-art performance. Detailed results can be found in the paper.

<details>
<summary>Click to expand</summary>




- quantitative comparisons in Table 3 (main paper)

<p align="center">
  <img width="900" src="figures/exp.png">
</p>



- visual comparison in Figure 1 (main paper)

<p align="center">
  <img width="900" src="figures/comp1.png">
</p>



- visual comparison in Figure 6 (main paper)

<p align="center">
  <img width="900" src="figures/comp2.png">
</p>




- visual comparison in Figure 12 (supplemental material)

<p align="center">
  <img width="900" src="figures/comp3.png">
</p>

</details>



## <a name="acknowledgements"></a> Acknowledgements

This code is built on [BasicSR](https://github.com/XPixelGroup/BasicSR).
