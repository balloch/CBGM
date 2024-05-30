# CBGM
An asynchronous fork of the original code for [Concept Bottleneck Generative Models](https://openreview.net/forum?id=L9U5MJJleF)

If you use any of the code or work in this repo or the associated paper please attribute to the original authors

## Installation

 Dependencies can be installed using the following command:

```bash
pip install -r requirements.txt
```

## Usage
Create color-MNIST Dataset:

```bash
python create_dataset.py
```

Commands for training GAN without CB on color-MNIST:

```bash
python train/train_gan.py
```

Commands for training GAN with CB on color-MNIST:


```bash
python train/train_cb_gan.py
```

BibTeX:
```
@inproceedings{
ismail2024concept,
title={Concept Bottleneck Generative Models},
author={Aya Abdelsalam Ismail and Julius Adebayo and Hector Corrada Bravo and Stephen Ra and Kyunghyun Cho},
booktitle={The Twelfth International Conference on Learning Representations},
year={2024},
url={https://openreview.net/forum?id=L9U5MJJleF}
}
```
