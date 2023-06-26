# Meta-Pec
The implementation of Meta-Pec.

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
