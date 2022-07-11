# HOP: History-and-Order Aware Pre-training for Vision-and-Language Navigation

This repository is the official implementation of [HOP: History-and-Order Aware Pre-training for Vision-and-Language Navigation](https://arxiv.org/abs/2203.11591). 

![architecture](files/architecture.jpg)

## Prerequisites
```sh
# Set up with Anaconda
conda env create -f hop_env.yaml
conda activate hop
```
## Quick Start
1. Download processed data and pretrained models.
   Please follow the instructions below to prepare the data in directories:
   - MP3D navigability graphs: `connectivity`
     - Download the [connectivity maps ](https://github.com/peteanderson80/Matterport3DSimulator/tree/master/connectivity).
   - Processed data: `data`
     - Download the [Processed data](https://github.com/YanyuanQiao/HOP-VLN/tree/main/data).

2. Run Pre-training
   ```sh
   bash run/pretrain.bash
   ```
   The trained model will be saved under `result/`.
   
   You can also train model using only the processed PREVALENT data:
   
   let `--prevalent_only = True` in `pretrain.bash`
   
3. Run finetuning
   * Please check [here](https://github.com/YanyuanQiao/HOP-VLN-finetune) for experiment setup and HOP application.

## Citation
If you use or discuss our Recurrent VLN-BERT, please cite our paper:
```
@InProceedings{Qiao2022HOP,
    author    = {Hong, Yicong and Wu, Qi and Qi, Yuankai and Rodriguez-Opazo, Cristian and Gould, Stephen},
    title     = {A Recurrent Vision-and-Language BERT for Navigation},
    booktitle = {Proceedings of the IEEE/CVF Conference on Computer Vision and Pattern Recognition (CVPR)},
    month     = {June},
    year      = {2021},
    pages     = {1643-1653}
}
```
