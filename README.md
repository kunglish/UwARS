# Experimental Subjects: Datasets and Models

This document describes the datasets and DNN models used in the experiments of **UwARS** (Uncertainty-weighted Adaptive Random Selection for DNN test selection). Our experimental subjects follow those of SETS (Wang et al., TOSEM 2026); the dataset sources and pretrained models below are consistent with the official SETS replication package: <https://github.com/GIST-NJU/SETS>.

## 1. Datasets

We use six widely adopted image classification datasets, covering grayscale and color images, simple and complex scenes, and 10 to 200 classes.

| Dataset | Classes | Training Set | Test Set | Image Size | Source |
|---|---|---|---|---|---|
| MNIST | 10 | 60,000 | 10,000 | 28×28 (grayscale) | `keras.datasets.mnist` |
| Fashion-MNIST | 10 | 60,000 | 10,000 | 28×28 (grayscale) | `keras.datasets.fashion_mnist` |
| CIFAR-10 | 10 | 50,000 | 10,000 | 32×32×3 | `keras.datasets.cifar10` |
| SVHN | 10 | 73,257 | 26,032 | 32×32×3 | <http://ufldl.stanford.edu/housenumbers/> |
| Fruits-360 | 141 | 70,589 | 23,619 | 100×100×3 | <https://github.com/fruits-360/fruits-360-100x100> (100×100 version) |
| TinyImageNet | 200 | 100,000 | 10,000 | 64×64×3 | <https://www.kaggle.com/c/tiny-imagenet/data> |

Notes:
- **MNIST, Fashion-MNIST, and CIFAR-10** are loaded directly through the Keras API (`keras.datasets`); no manual download is required.

## 2. DNN Models

Seven DNN models are paired with the six datasets, forming **eight dataset–model combinations (subjects)**. The architectures range from shallow LeNet-style networks to deep residual networks.

| # | Dataset | Model | Framework | Pretrained Model Source |
|---|---|---|---|---|
| 1 | MNIST | LeNet-1 | Keras/TensorFlow | [DeepGD repository (Org_model)](https://github.com/ZOE-CA/DeepGD/tree/main/Retraining/Org_model) |
| 2 | MNIST | LeNet-5 | Keras/TensorFlow | [DeepGD repository (Org_model)](https://github.com/ZOE-CA/DeepGD/tree/main/Retraining/Org_model) |
| 3 | CIFAR-10 | 12-layer ConvNet | Keras/TensorFlow | [DeepGD repository (Org_model)](https://github.com/ZOE-CA/DeepGD/tree/main/Retraining/Org_model) |
| 4 | CIFAR-10 | ResNet-20 | Keras/TensorFlow | [DeepGD repository (Org_model)](https://github.com/ZOE-CA/DeepGD/tree/main/Retraining/Org_model) |
| 5 | Fashion-MNIST | LeNet-4 | Keras/TensorFlow | [DeepGD repository (Org_model)](https://github.com/ZOE-CA/DeepGD/tree/main/Retraining/Org_model) |
| 6 | SVHN | LeNet-5 | Keras/TensorFlow | [DeepGD repository (Org_model)](https://github.com/ZOE-CA/DeepGD/tree/main/Retraining/Org_model) |
| 7 | Fruits-360 | ResNet-50 | Keras/TensorFlow | Provided in the `Pretrained_model` folder of the [SETS replication package](https://github.com/GIST-NJU/SETS) |
| 8 | TinyImageNet | ResNet-101 | PyTorch | [Google Drive](https://drive.google.com/drive/folders/1RLyQIcJ8qNqds9US-Oo2a0uQEL0t6kSZ) |
