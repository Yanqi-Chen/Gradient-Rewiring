# Pruning of Deep Spiking Neural Networks through Gradient Rewiring
This directory contains the code of this paper. The pretrained model is too large to fit <50MB requirements for supplementary file. Nonetheless, we spare no effort to maintain the reproducibility by keeping the random seeds in our experiment and clarifying the dependency and environment.

- [Directory Tree](#directory-tree)
- [Dependency](#dependency)
- [Environment](#environment)
- [Usage](#usage)
- [Running Arguments](#running-arguments)
- [Citation](#citation)

## Directory Tree

```
.
├── c10
│   ├── c10.py
│   ├── __init__.py
│   └── model.py
├── deeprewire.py
├── gradrewire.py
├── mnist
│   ├── __init__.py
│   ├── mnist.py
│   └── model.py
└── README.md
```

The training (including test) code and model definition for CIFAR-10 and MNIST are located on corresponding two separate directory (c10 and mnist). The proposed *Grad Rewiring* algorithm is integrated with Adam optimizer in file `gradrewire.py` as a PyTorch optimizer. The code of *Deep Rewiring algorithm (Deep R)* is organized in the same way. 

## Dependency 

The major dependencies of this repo are list as below

```
# Name                    Version
cudatoolkit               10.1.243
cudnn                     7.6.5
numpy                     1.19.1
python                    3.7.9 
pytorch                   1.6.0
spikingjelly              <Specific Version>
tensorboard               2.2.1
torchvision               0.7.0
```

**Note**: the version of spikingjelly will be clarified in [usage](#usage) part.

## Environment

The code requires NVIDIA GPU and has been tested on *CUDA 10.1* and *Ubuntu 16.04*. You may need GPU with **>6GB video memory** to get the code run as the same batch size in our paper to reproduce the results.

- GPU: Tesla *V100-SXM2-32GB* 300 Watts version
- CPU: Intel(R) Xeon(R) Platinum 8268 CPU @ 2.90GHz

We use a single Tesla V100 GPU for each experiment. We recommend GPU with ECC enabled if you want exactly the same results (e.g. the training curves shown in paper).

### Epoch Time (Wall Clock Time)

The rough running time here is measured on platforms mentioned above and should only be regarded as a reference.

| Dataset  | Train & Test (s) | Train Only (s) |
| -------- | ---------------- | -------------- |
| CIFAR-10 | 1150             | 540            |
| MNIST    | 12.8             | 12.3           |

There are many intricate data processing in the test stage, consuming much time.

## Usage

This code requires a *legacy* version of an open-source SNN framework **SpikingJelly**. To get this framework installed, first clone the repo from [GitHub](https://github.com/fangwei123456/spikingjelly):

```bash
$ git clone https://github.com/fangwei123456/spikingjelly.git
```

or [OpenI](https://git.openi.org.cn/OpenI/spikingjelly):

```bash
$ git clone https://git.openi.org.cn/OpenI/spikingjelly.git
```

Then, checkout the version we use in these experiments and install it.

```bash
$ cd spikingjelly
$ git checkout c8a9ba8
$ python setup.py install
```

With dependency mentioned above installed, you should be able to run the following commands:

### Grad Rewiring on CIFAR-10:

```shell
$ cd <repo_path>/c10
$ python c10.py -s 0.95 -gpu <gpu_id> --dataset-dir <dataset_path> --dump-dir <dump_logs&models_path> -m grad
```

### Grad Rewiring on MNIST:

```shell
$ cd <repo_path>/mnist
$ python mnist.py -s 0.95 -gpu <gpu_id> --dataset-dir <dataset_path> --dump-dir <dump_logs&models_path> -m grad
```

The TensorBoard logs will be placed in `<dump-dir>/logs`.

## Running Arguments

| Arguments           | Descriptions                                                 | Default Value             | Type  |
| ------------------- | ------------------------------------------------------------ | ------------------------- | ----- |
| -b,--batch-size     | Training batch size                                          | 128(MNIST),16(CIFAR-10)   | int   |
| -lr,--learning-rate | Learning rate                                                | 1e-4                      | float |
| -penalty            | L1 penalty for Deep R, prior term for Grad Rewiring          | 1e-3                      | float |
| -s,--sparsity       | Maximum sparsity for Deep R, target sparsity for soft-Deep R and Grad Rewiring |                           | float |
| -gpu                | GPU id                                                       |                           | str   |
| --dataset-dir       | Path of datasets                                             |                           | str   |
| --dump-dir          | Path for dumping models and logs                             |                           | str   |
| -T                  | Simulation time-steps                                        | 8                         | int   |
| -N,--epoch          | Number of training epochs                                    | 512(MNIST),2048(CIFAR-10) | int   |
| -m,--mode           | Pruning method ('deep' or  'grad', or 'no_prune')            | 'no_prune'                | str   |
| -soft               | Whether to use soft Deep R (Only work when mode='deep')      | False                     | bool  |
| -test               | Whether to test only                                         | False                     | bool  |

## Citation

Please refer to the following citation if this work is useful for your research.

```latex
@inproceedings{ijcai2021-236,
  title     = {Pruning of Deep Spiking Neural Networks through Gradient Rewiring},
  author    = {Chen, Yanqi and Yu, Zhaofei and Fang, Wei and Huang, Tiejun and Tian, Yonghong},
  booktitle = {Proceedings of the Thirtieth International Joint Conference on
               Artificial Intelligence, {IJCAI-21}},
  publisher = {International Joint Conferences on Artificial Intelligence Organization},
  editor    = {Zhi-Hua Zhou},
  pages     = {1713--1721},
  year      = {2021},
  month     = {8},
  note      = {Main Track}
  doi       = {10.24963/ijcai.2021/236},
  url       = {https://doi.org/10.24963/ijcai.2021/236},
}
```

