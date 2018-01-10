# Pytorch-pensieve
This is a pytorch implementation of [pensieve](https://github.com/shinshiner/Pytorch-pensieve#reference). The environment and system are based on the original version and the reinforcement learning algorithm is highly brought from [rl_a3c_pytorch](https://github.com/shinshiner/Pytorch-pensieve#reference). The network architecture is totoally changed to get a better result.

## Requirements
#### Main Framework
* Ubuntu14.04

* Python2.7 & Python3.4(5)

* [Pytorch](http://pytorch.org/)

#### Python Packages
* selenium

* pyvirtualdisplay

#### Other Dependences
* apache2

* Google Chrome browser

You can install parts of them by running `python setup.py` and `python3 setup.py`.

## To Get Data
Run `cd data` and follow the [Readme.md](https://github.com/shinshiner/Pytorch-pensieve/blob/master/data/Readme.md) in that folder.

## Training
Run `cd train` and follow the [Readme.md](https://github.com/shinshiner/Pytorch-pensieve/blob/master/train/Readme.md) in that folder.

## Testing
Run `cd test` and follow the [Readme.md](https://github.com/shinshiner/Pytorch-pensieve/blob/master/test/Readme.md) in that folder.

## Real Experiment
To run real experiment, you need to install the dependences in python2.7 first.

1) Modify the `NN_MODEL` in `rl_server/rl_server.py`
2) `cd exp`
3) `python run_exp.py`

## Reference
* [Pensieve](https://github.com/hongzimao/pensieve)

* [rl_a3c_pytorch](https://github.com/dgriff777/rl_a3c_pytorch)
