# 2DQuant: Low-bit Post-Training Quantization for Image Super-Resolution

[Kai Liu](https://kai-liu001.github.io/), [Haotong Qin](https://htqin.github.io/), [Yong Guo](https://www.guoyongcs.com/), [Xin Yuan](https://en.westlake.edu.cn/faculty/xin-yuan.html), [Linghe Kong](https://www.cs.sjtu.edu.cn/~linghe.kong/), [Guihai Chen](https://cs.nju.edu.cn/gchen/index.htm), and [Yulun Zhang](http://yulunzhang.com/), "2DQuant: Low-bit Post-Training Quantization for Image Super-Resolution", arXiv, 2024

[[arXiv](https://arxiv.org/abs/2406.06649)] [visual results] [pretrained models]



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


## Dependencies

- Python 3.8
- PyTorch 1.8.0
- NVIDIA GPU + [CUDA](https://developer.nvidia.com/cuda-downloads)

```bash
# Clone the github repo and go to the default directory 'DAT'.
git clone https://github.com/Kai-Liu001/2DQuant.git
conda create -n tdquant python=3.8
conda activate tdquant
pip install -r requirements.txt
python setup.py develop
```


## Contents

1. [Datasets](#datasets)
1. [Models](#models)
1. [Training](#training)
1. [Testing](#testing)
1. [Results](#results)
1. [Citation](#citation)
1. [Acknowledgements](#acknowledgements)

## <a name="datasets"></a> Datasets

Used training and testing sets can be downloaded as follows:

| Training Set                                                 |                         Testing Set                          |                        Visual Results                        |
| :----------------------------------------------------------- | :----------------------------------------------------------: | :----------------------------------------------------------: |
| [DIV2K](https://data.vision.ee.ethz.ch/cvl/DIV2K/) (800 training images, 100 validation images) +  [Flickr2K](https://cv.snu.ac.kr/research/EDSR/Flickr2K.tar) (2650 images) [complete training dataset DF2K: [Google Drive](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link) / [Baidu Disk](https://pan.baidu.com/s/1KIcPNz3qDsGSM0uDKl4DRw?pwd=74yc)] | Set5 + Set14 + BSD100 + Urban100 + Manga109 [complete testing dataset: [Google Drive](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) / [Baidu Disk](https://pan.baidu.com/s/1Tf8WT14vhlA49TO2lz3Y1Q?pwd=8xen)] | [Google Drive](https://drive.google.com/drive/folders/1ZMaZyCer44ZX6tdcDmjIrc_hSsKoMKg2?usp=drive_link) / [Baidu Disk](https://pan.baidu.com/s/1LO-INqy40F5T_coAJsl5qw?pwd=dqnv#list/path=%2F) |

Download training and testing datasets and put them into the corresponding folders of `datasets/`.

## <a name="models"></a>Models

The pretrained models will be released soon.

## <a name="training"></a> Training
Training is used to optimize the quantizers' parameters.

- Download [training](https://drive.google.com/file/d/1TubDkirxl4qAWelfOnpwaSKoj3KLAIG4/view?usp=share_link) (DF2K, already processed) and [testing](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) (Set5, BSD100, Urban100, Manga109, already processed) datasets, place them in `datasets/`.
- Download [cali data]() and place them in `keydata/` or run `scripts/2DQuant-getcalidata.sh` to obtain `calidata`.

- Run the following scripts. The training configuration is in `options/train/`. More scripts can be found in `scripts/2DQuant-train.sh`.

  ```shell
  # 2DQuant 4bit x4
  python basicsr/train.py -opt options/train/train_2DQuant_x4.yml --force_yml bit=4 name=train_2DQuant_x4_bit4
  ```
  
- The training experiment is in `experiments/`.


## <a name="testing"></a> Testing

- Download the pre-trained [models]() (pre-trained models will be released soon.) and place them in `experiments/pretrained_models/`.

- Download [testing](https://drive.google.com/file/d/1yMbItvFKVaCT93yPWmlP3883XtJ-wSee/view?usp=sharing) (Set5, BSD100, Urban100, Manga109) datasets, place them in `datasets/`.

- Run the following scripts. The testing configuration is in `options/test/`.

  ```shell
  # 2DQuant, reproduces results in Table 3 of the main paper
  python basicsr/test.py -opt options/test/test_2DQuant_x2.yml --force_yml bit=4 name=test_2DQuant_x2_bit4 path:pretrain_network_Q=experiments/train_2DQuant_x2_bit4/models/net_Q_3200.pth 
  ```
  
- The output is in `results/`.

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
