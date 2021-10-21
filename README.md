<div align="center">
  <img width="75%" alt="QueryInst-VIS Demo" src="https://user-images.githubusercontent.com/45201863/138304424-41279aa5-86b0-4c1c-b747-4f1788f78d7a.png">
</div>

<div align="center">
  <img width="75%" alt="QueryInst-VIS Demo" src="https://user-images.githubusercontent.com/45201863/138265556-7c58cd89-d0a9-4708-b37f-e9f166443c92.gif">
</div>

* **TL;DR:** **CrossVIS (Crossover Learning for Fast Online Video Instance Segmentation)** proposes a novel crossover learning paradigm to fully leverage rich contextual information across video frames, and obtains great trade-off between accuracy and speed for video instance segmentation.

# Crossover Learning for Fast Online Video Instance Segmentation

</br>

> [**Crossover Learning for Fast Online Video Instance Segmentation**](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Crossover_Learning_for_Fast_Online_Video_Instance_Segmentation_ICCV_2021_paper.pdf) (ICCV 2021)
>
> by [Shusheng Yang\*](https://scholar.google.com/citations?hl=zh-CN&user=v6dmW5cntoMC&view_op=list_works&citft=1&email_for_op=2yuxinfang%40gmail.com&gmla=AJsN-F53CnxYBtSUBs91e_N7uL7139t5ufTWFZ-r8k5oNe1haqf_6f8AE0uyoqnVBPqNG8MGOPH_ep6k_-gMW9KmflOUalJPYu1VTaE2IVjNVn1k-lDjzMEN_oN_a7MySKPieyFEPwMfabczLcR4Qg14seBM3mx6QXUu9Hj5QMZrg9jbKDOGQxxeVX0DJtjiWCGr2ukQgSIR4VVetSaGei48SNUkO8zol-8hApyNYZcUBLD6n9FvTEeE94iLiIbFbNP0m59fh3_z), [Yuxin Fang\*](https://scholar.google.com/citations?user=_Lk0-fQAAAAJ&hl=en), [Xinggang Wang†](https://xinggangw.info/), [Yu Li](http://yu-li.github.io), [Chen Fang](https://scholar.google.com/citations?hl=en&user=Vu1OqIsAAAAJ&view_op=list_works&citft=1&email_for_op=2yuxinfang%40gmail.com&gmla=AJsN-F5phq2a5UjdoNudoavuaCbem43ptau5cM8rWScWoxkUm0xFgCl6q49r-6MAWh-X9FVZCv9GuLk8D4u-ka0hVjKEWibox_kN9B346lA80Mrl4bUyDHBjwmbvsAfEBZ56neZ0D9p5neQBX8dBp8dD1I3248R0n0vVvzlfILA9oVpcn7xy6P0MQHUY-g0VT2g7sV6LJSPB7ZGyJFGqUk2SJ4MHRxG8U7Hz28WGuobOz-lrTnehfz5wsbwAaLETSZbP3vEMQ3Hc), [Ying Shan](https://scholar.google.com/citations?user=4oXBp9UAAAAJ&hl=en), Bin Feng, [Wenyu Liu](http://eic.hust.edu.cn/professor/liuwenyu/).
>
> (\*) equal contribution, (†) corresponding author.
>
> *[ICCV2021 Paper](https://openaccess.thecvf.com/content/ICCV2021/papers/Yang_Crossover_Learning_for_Fast_Online_Video_Instance_Segmentation_ICCV_2021_paper.pdf)*

</br>

<div align="center">
  <img width="100%" alt="QueryInst-VIS Demo" src="https://user-images.githubusercontent.com/45201863/138266370-0fe4cb3e-74dc-4a55-90c9-76f92a78e548.gif">
</div>

## Main Results on YouTube-VIS 2019 Dataset

* We provide both checkpoints and codalab server submissions in the bellow link.

Name | AP | AP@50 | AP@75 | AR@1 | AR@10 | download
--- |:---:|:---:|:---:|:---:|:---:|:---:
[CrossVIS_R_50_1x](configs/CrossVIS/R_50_1x.yaml) | 35.5 | 55.1 | 39.0 | 35.4 | 42.2 | [baidu](https://pan.baidu.com/s/10ccx2RrA-TCJr8ZNXmlm2w)(keycode: ```a0j0```) &#124; [google](https://drive.google.com/file/d/1qDFf-Fk2R67iyzwR1_WbZ3IfwYgAR5vu/view?usp=sharing, https://drive.google.com/file/d/1zuBu4XpPlbCjh9jM00jPb5phk5jdl4la/view?usp=sharing)
[CrossVIS_R_101_1x](configs/CrossVIS/R_101_1x.yaml) | 36.9 | 57.8 | 41.4 | 36.2 | 43.9 | [baidu](https://pan.baidu.com/s/1orXUtSQC_1ZvsOa6XcFteA)(keycode: ```iwwo```) &#124; [google](https://drive.google.com/file/d/11BRxiLDdqreEI66Lp0EwPvnHZn8msraa/view?usp=sharing, https://drive.google.com/file/d/1GGwUkqs6HhM4GHRkxQPXppun6QyvR8oh/view?usp=sharing)

## Getting Started

### Installation

First, clone the repository locally:

```bash
git clone https://github.com/hustvl/CrossVIS.git
```

Then, create python virtual environment with conda:
```bash
conda create --name crossvis python=3.7.2
conda activate crossvis
```

Install torch 1.7.0 and torchvision 0.8.1:
```bash
pip install torch==1.7.0 torchvision==0.8.1
```

Follow the [instructions](https://detectron2.readthedocs.io/en/latest/tutorials/install.html) to install ```detectron2```. Please install ```detectron2``` with commit id [9eb4831](https://github.com/facebookresearch/detectron2/commit/9eb4831f742ae6a13b8edb61d07b619392fb6543) if you have any issues related to ```detectron2```.

Then install ```AdelaiDet``` by:
```bash
cd CrossVIS
python setup.py develop
```

### Preparation

* Download ```YouTube-VIS 2019``` dataset from [here](https://youtube-vos.org/dataset/vis/), the overall directory hierarchical structure is:
```
CrossVIS
├── datasets
│   ├── youtubevis
│   │   ├── train
│   │   │   ├── 003234408d
│   │   │   ├── ...
│   │   ├── val
│   │   │   ├── ...
│   │   ├── annotations
│   │   │   ├── train.json
│   │   │   ├── valid.json
```
* Download ```CondInst``` 1x pretrained model from [here](https://github.com/aim-uofa/AdelaiDet/blob/master/configs/CondInst/README.md)

### Training

* Train CrossVIS R-50 with single GPU:
```bash
python tools/train_net.py --config configs/CrossVIS/R_50_1x.yaml MODEL.WEIGHTS $PATH_TO_CondInst_MS_R_50_1x
```

* Train CrossVIS R-50 with multi GPUs:
```bash
python tools/train_net.py --config configs/CrossVIS/R_50_1x.yaml --num-gpus $NUM_GPUS MODEL.WEIGHTS $PATH_TO_CondInst_MS_R_50_1x
```

### Inference

```bash
python tools/test_vis.py --config-file configs/CrossVIS/R_50_1x.yaml --json-file datasets/youtubevis/annotations/valid.json --opts MODEL.WEIGHTS $PATH_TO_CHECKPOINT
```

The final results will be stored in ```results.json```, just compress it with ```zip``` and upload to the codalab server to get the performance on validation set.

## Acknowledgement :heart:

This code is mainly based on [```detectron2```](https://github.com/facebookresearch/detectron2.git) and [```AdelaiDet```](https://github.com/aim-uofa/AdelaiDet.git), thanks for their awesome work and great contributions to the computer vision community!

## Citation

If you find our paper and code useful in your research, please consider giving a star :star: and citation :pencil: :

```BibTeX
@InProceedings{Yang_2021_ICCV,
    author    = {Yang, Shusheng and Fang, Yuxin and Wang, Xinggang and Li, Yu and Fang, Chen and Shan, Ying and Feng, Bin and Liu, Wenyu},
    title     = {Crossover Learning for Fast Online Video Instance Segmentation},
    booktitle = {Proceedings of the IEEE/CVF International Conference on Computer Vision (ICCV)},
    month     = {October},
    year      = {2021},
    pages     = {8043-8052}
}
```
