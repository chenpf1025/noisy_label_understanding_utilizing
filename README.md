# Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels
This is the Keras implementation for the ICML 2019 paper 'Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels'.


```
@inproceedings{chen2019understanding,
  title={Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels},
  author={Pengfei Chen, Benben Liao, Guangyong Chen, Shengyu Zhang},
  booktitle={ICML},
  year={2019}
}
```

## Dependencies
Python 3, Keras, Tensorflow, numpy, sklearn

## Setup
To set up experiments, we need to download the [CIFAR-10 data](https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz) and extract it to:

```
data/cifar-10-batches-py
```
The code will automatically add noise to CIFAR-10 by randomly flipping original labels.

## Understanding noisy labels
**Note**

To quantitatively characterize the generalization performance of deep neural networks normally trained with noisy labels, we split the noisy dataset into two halves and perform cross-validation: training on a subset and testing on the other.

We firstly theoretically characterize on the test set the ```confusion matrix (w.r.t. ground-truth labels)``` and ```test accuracy (w.r.t. noisy labels)```.

We then propose to select a testing sample as a clean one, if the trained model predict the same label with its observed label. The performance is evaluated by ```label precision``` and ```label recall```, which can be theoretically estimated using the noise ratio according to our paper.

**Train**

Experimrental resluts justify our theoretical analysis. To reproduce the experimental results, we can run ```Verify_Theory.py``` and specify the noise pattern and noise ratio, e.g., 

* Symmetric noise with ratio 0.5:

  ```python Verify_Theory.py --noise_pattern sym --noisy_ratio 0.5```
  
* Asymmetric noise with ratio 0.4:

  ```python Verify_Theory.py --noise_pattern asym --noisy_ratio 0.4``` 

**Results**

Test accuracy, label precision and label recall w.r.t noise ratio on manually corrupted CIFAR-10. 

![image](https://github.com/chenpf1025/noisy_label_understanding_utilizing/blob/master/results/LPLR.png)

Confusion matrix M approximates noise transistion matrix T.

![image](https://github.com/chenpf1025/noisy_label_understanding_utilizing/blob/master/results/Confusion.png)

## Identifying clean labels and robustly train deep neural networks
**Note**

We present the Iterative Noisy Cross-Validation (INCV) to select a subset of clean samples, then modify the [Co-teaching](https://arxiv.org/abs/1804.06872) strategy to train noise-robust deep neural networks. 

**Train**

E.g., use our method to train on CIFAR-10 with

* 50% symmetric noise:

  ```python INCV_main.py --noise_pattern sym --noisy_ratio 0.5 --dataset cifar10```
  
* 40% asymmetric noise:

  ```python INCV_main.py --noise_pattern asym --noisy_ratio 0.4 --dataset cifar10```

**Results**

Average test accuracy (%, 5 runs) with standard deviation:

|    Method    |  Sym. 0.2  |   Sym. 0.5  |  Sym. 0.8  |  Aym. 0.4  |
|:------------:|:----------:|:-----------:|:----------:|:----------:|
| [F-correction](https://arxiv.org/abs/1609.03683) | 85.08±0.43 |  76.02±0.19 | 34.76±4.53 | 83.55±2.15 |
|  [Decoupling](https://arxiv.org/abs/1706.02613)  | 86.72±0.32 |  79.31±0.62 | 36.90±4.61 | 75.27±0.83 |
|  [Co-teaching](https://arxiv.org/abs/1804.06872) | 89.05±0.32 |  82.12±0.59 | 16.21±3.02 | 84.55±2.81 |
|   [MentorNet](https://arxiv.org/abs/1712.05055)  | 88.36±0.46 |  77.10±0.44 | 28.89±2.29 | 77.33±0.79 |
|      [D2L](https://arxiv.org/abs/1806.02612)     | 86.12±0.43 | 67.39±13.62 | 10.02±0.04 | 85.57±1.21 |
|     Ours     | 89.71±0.18 |  84.78±0.33 | 52.27±3.50 | 86.04±0.54 |

Average test accuracy (%, 5 runs) during training:

![image](https://github.com/chenpf1025/noisy_label_understanding_utilizing/blob/master/results/TestAcc.png)

## Cite
Please cite our paper if you use this code in your research work.

## Questions/Bugs
Please submit a Github issue or contact pfchen@cse.cuhk.edu.hk if you have any questions or find any bugs.
