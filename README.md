# Learning the Latent Causal Structure for Modeling Label Noise

Official implementation of Learning the Latent Causal Structure for Modeling Label Noise (NeurIPS 2024).

## Abstract

In label-noise learning, the noise transition matrix reveals how an instance transitions from its clean label to its noisy label. Accurately estimating an instance's noise transition matrix is crucial for estimating its clean label. However, when only a noisy dataset is available, noise transition matrices can be estimated only for some "special" instances. To leverage these estimated transition matrices to help estimate the transition matrices of other instances, it is essential to explore relations between the matrices of these "special" instances and those of others. Existing studies typically build the relation by explicitly defining the similarity between the estimated noise transition matrices of "special" instances and those of other instances. However, these similarity-based assumptions are hard to validate and may not align with real-world data. If these assumptions fail, both noise transition matrices and clean labels cannot be accurately estimated. In this paper, we found that by learning the latent causal structure governing the generating process of noisy data, we can estimate noise transition matrices without the need for similarity-based assumptions. Unlike previous generative label-noise learning methods, we consider causal relations between latent causal variables and model them with a learnable graphical model. Utilizing only noisy data, our method can effectively learn the latent causal structure. Experimental results on various noisy datasets demonstrate that our method achieves state-of-the-art performance in estimating noise transition matrices, which leads to improved classification accuracy.

## Experiments

Environment: Python 3.9

To install the necessary Python packages:

```
pip install -r requirements.txt
```

Download the CIFAR-10 and CIFAR-100 datasets:

```
wget -c https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz
tar -xzvf cifar-10-python.tar.gz
mv cifar-10-batches-py cifar-10
wget -c https://www.cs.toronto.edu/~kriz/cifar-100-python.tar.gz
tar -xzvf cifar-100-python.tar.gz
mv cifar-100-batches-py cifar-100
```

For the FashionMNIST dataset, we use the dataset provided by [PTD](https://github.com/xiaoboxia/Part-dependent-label-noise). The images and labels have been processed to .npy format. You can download the [fashionmnist](https://drive.google.com/open?id=1Tz3W3JVYv2nu-mdM6x33KSnRIY1B7ygQ) here.

Before training the model, warmup the model and obtain the checkpoint.

Example on CIFAR-10:

```
python warmup.py --noise_mode instance --dataset cifar10 --data_path ./cifar-10 --num_class 10 --r 0.5
```

Then, train the model.

Example on CIFAR-10:

```
python Train_fmnist_cifar.py --noise_mode instance --dataset cifar10 --data_path ./cifar-10 --num_class 10 --r 0.5
```

To train the model on CIFARN:

```
python Train_fmnist_cifar.py --noise_mode worse_label --dataset cifar10 --data_path ./cifar-10 --num_class 10 --noise_mode worse_label --lambda_u 20 --lambda_elbo 0.1
```

## Citation

If you find our work insightful, please consider citing our paper:

```
@inproceedings{
lin2024learning,
title={Learning the Latent Causal Structure for Modeling Label Noise},
author={Yexiong Lin and Yu Yao and Tongliang Liu},
booktitle={Thirty-eighth Conference on Neural Information Processing Systems},
year={2024}
}
```

