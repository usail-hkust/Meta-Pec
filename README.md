# Meta-Pec
This is the official implementation of [A Preference-aware Meta-optimization Framework for Personalized Vehicle Energy Consumption Estimation](https://arxiv.org/abs/2306.14421) from KDD 2023.

## Environment
* Python=3.7
* PyTorch=1.13.0
* numpy=1.21.6
* learn2learn=0.1.7

## Datasets
* **VED**: The first is a large-scale energy usage dataset of diverse vehicles in Ann Arbor, Michigan, USA, known as the vehicle energy dataset ([VED](https://github.com/gsoh/VED)).
* **ETTD**: Another is an electric taxi trajectory dataset ([ETTD](http://guangwang.me/\#/data)), collected from Shenzhen, Guangdong, China.

## Usage
Our data has been preprocessed and is available at https://www.dropbox.com/s/h2bnavg09gcloet/datasets.zip?dl=0. You need to download the *datasets* folder and put it under the root.

### Train and test

Train and evaluate the model:
```sh
python main.py --dataset VED
```

### Citation
```
@inproceedings{DBLP:conf/kdd/LaiZL23,
  author       = {Siqi Lai and
                  Weijia Zhang and
                  Hao Liu},
  editor       = {Ambuj Singh and
                  Yizhou Sun and
                  Leman Akoglu and
                  Dimitrios Gunopulos and
                  Xifeng Yan and
                  Ravi Kumar and
                  Fatma Ozcan and
                  Jieping Ye},
  title        = {A Preference-aware Meta-optimization Framework for Personalized Vehicle
                  Energy Consumption Estimation},
  booktitle    = {Proceedings of the 29th {ACM} {SIGKDD} Conference on Knowledge Discovery
                  and Data Mining, {KDD} 2023, Long Beach, CA, USA, August 6-10, 2023},
  pages        = {4346--4356},
  publisher    = {{ACM}},
  year         = {2023},
  url          = {https://doi.org/10.1145/3580305.3599767},
  doi          = {10.1145/3580305.3599767},
  timestamp    = {Thu, 24 Aug 2023 14:07:33 +0200},
  biburl       = {https://dblp.org/rec/conf/kdd/LaiZL23.bib},
  bibsource    = {dblp computer science bibliography, https://dblp.org}
}
```
