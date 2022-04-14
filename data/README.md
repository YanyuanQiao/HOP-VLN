## Pre-training Datasets
Our pre-training dataset based on existing datasets: `PREVALENT` and `BnB`. `PREVALENT` uses a pre-trained speaker model to produce more instructions to augment R2R dataset. To better capture order information from the trajectory, we use the front view image of the agent’s observation at each position, rather than using the panoramic image. `BnB` collects image-caption pairs from Airbnb. We use raw images and captions from the BnB dataset and reprocessed them.

### Processed PREVALENT data
- Download the [processed PREVALENT data](https://drive.google.com/drive/folders/1jyaHqqOk2P9AKgh1EMx6dqqTWOsnGeo5?usp=sharing).
### Processed BnB data
Follow instructions in [bnb-dataset](https://github.com/airbert-vln/bnb-dataset) to download listing from Airbnb and images.
The process instruction is coming soon!

- After downloading processed PREVALENT data and processing BnB data you should see the following folder structure:
```
    ├── data
    │   └── train
    │       └── short_generated.json
    │   ├── neg
    │       ├── bnb_neg.json
    │       ├── candidtae_sm_seq.json
    │       └── scan_candidate_sm.json
    │   ├── bnb
    │       ├── traj_train.json
    │       └── traj_test.json
    │   └── collect_traj_test
    └── img_features
        ├── ResNet-152-imagenet.tsv
        └── bnbdata.npz
```
- `short_generated.json` and `ResNet-152-imagenet.tsv` are same as PREVALENT.
- `candidtae_sm_seq.json` and `scan_candidate_sm.json` are negatives for processed PREVALENT data.
- `traj_train.json` and `traj_test.json` are training and test data for processed BnB data.
- `bnb_neg.json` is negatives for processed BnB data.

You can also train model using only the processed PREVALENT data. Please see https://github.com/YanyuanQiao/HOP-VLN/blob/0917125464240c37226d63342311d0e7bb4e7cc4/tasks/pretrain/main.py#L713

