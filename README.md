# Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels
This is a Keras implementation for the paper [Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels](http://proceedings.mlr.press/v97/chen19g.html)(Proceedings of ICML, 2019).


```
@inproceedings{chen2019understanding,
  title={Understanding and Utilizing Deep Neural Networks Trained with Noisy Labels},
  author={Chen, Pengfei and Liao, Ben Ben and Chen, Guangyong and Zhang, Shengyu},
  booktitle={International Conference on Machine Learning},
  pages={1062--1070},
  year={2019}
}
```

## Dependencies
Python 3.6.4, Keras 2.1.6, Tensorflow 1.7.0, numpy, sklearn.

Please be aware of the **bug** caused by different **versions** of Keras/tf. For example, in the callback functions in model.fit_generator, new keras versions use "val_accuracy" instead of "val_acc", for which you may not directly get an error but may fail to save the model. Please check the Documentation of Keras carefully if you use a different version.



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

  ```python Verify_Theory.py --noise_pattern sym --noise_ratio 0.5```
  
* Asymmetric noise with ratio 0.4:

  ```python Verify_Theory.py --noise_pattern asym --noise_ratio 0.4``` 

**Results**

Test accuracy, label precision and label recall w.r.t noise ratio on manually corrupted CIFAR-10. 

<div align=center><img src="https://github.com/chenpf1025/noisy_label_understanding_utilizing/blob/master/results/LPLR.png" width = "75%"/></div>

Confusion matrix M approximates noise transistion matrix T.

<div align=center><img src="https://github.com/chenpf1025/noisy_label_understanding_utilizing/blob/master/results/Confusion.png" width = "75%"/></div>

## Simply cleaning noisy datasets
**Train**

If you only want to use INCV to clean a noisy dataset, you can run INCV.py only, e.g., on CIFAR-10 with

* 50% symmetric noise:

  ```python INCV.py --noise_pattern sym --noise_ratio 0.5 --dataset cifar10```
  
* 40% asymmetric noise:

  ```python INCV.py --noise_pattern asym --noise_ratio 0.4 --dataset cifar10```
  
The results will be saved in 'results/(dataset)/(noise_pattern)/(noise_ratio)/(XXX.csv)' with columns ('y', 'y_noisy', 'select', 'candidate', 'eval_ratio').

**Results**

```label precision``` and ```label recall``` on the manually corrupted CIFAR-10.

<div align=center><img src="https://github.com/chenpf1025/noisy_label_understanding_utilizing/blob/master/results/INCV.png" width = "50%"/></div>

Our INCV accurately identifies most clean samples. For example, under symmetric noise of ratio 0.5, it selects about 90% (=LR) of the clean samples, and the noise ratio of the selected set is reduced to around 10% (=1−LP).

## Cleaning noisy datasets and robustly training deep neural networks
**Note**

We present the Iterative Noisy Cross-Validation (INCV) to select a subset of clean samples, then modify the [Co-teaching](https://arxiv.org/abs/1804.06872) strategy to train noise-robust deep neural networks. 

**Train**

E.g., use our method to train on CIFAR-10 with

* 50% symmetric noise:

  ```python INCV_main.py --noise_pattern sym --noise_ratio 0.5 --dataset cifar10```
  
* 40% asymmetric noise:

  ```python INCV_main.py --noise_pattern asym --noise_ratio 0.4 --dataset cifar10```

**Results**

Average test accuracy (%, 5 runs) with standard deviation:

|    Method    |  Sym. 0.2  |   Sym. 0.5  |  Sym. 0.8  |  Aym. 0.4  |
|:------------:|:----------:|:-----------:|:----------:|:----------:|
| [F-correction](https://arxiv.org/abs/1609.03683) | 85.08±0.43 |  76.02±0.19 | 34.76±4.53 | 83.55±2.15 |
|  [Decoupling](https://arxiv.org/abs/1706.02613)  | 86.72±0.32 |  79.31±0.62 | 36.90±4.61 | 75.27±0.83 |
|  [Co-teaching](https://arxiv.org/abs/1804.06872) | 89.05±0.32 |  82.12±0.59 | 16.21±3.02 | 84.55±2.81 |
|   [MentorNet](https://arxiv.org/abs/1712.05055)  | 88.36±0.46 |  77.10±0.44 | 28.89±2.29 | 77.33±0.79 |
|      [D2L](https://arxiv.org/abs/1806.02612)     | 86.12±0.43 | 67.39±13.62 | 10.02±0.04 | 85.57±1.21 |
|     Ours     | **89.71±0.18** |  **84.78±0.33** | **52.27±3.50** | **86.04±0.54** |


Average test accuracy (%, 5 runs) during training:

<div align=center><img src="https://github.com/chenpf1025/noisy_label_understanding_utilizing/blob/master/results/TestAcc.png" width = "100%"/></div>

## Cite
Please cite our paper if you use this code in your research work.

## Questions/Bugs
Please submit a Github issue or contact chenpf.cuhk@gmail.com if you have any questions or find any bugs.
