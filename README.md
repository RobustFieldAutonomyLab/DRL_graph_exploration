# Autonomous Exploration Under Uncertainty via Deep Reinforcement Learning on Graphs
This repository contains code for robot exploration under uncertainty that uses graph neural networks (GNNs) in conjunction with deep reinforcement learning (DRL), enabling decision-making over graphs containing exploration information to predict a robot’s optimal sensing action in belief space. A demonstration video can be found [here](https://).

<p align='center'>
    <img src="/doc/exploration_graph.png" alt="drawing" width="1000"/>
</p>

<p float="left">
  <img src="/doc/test40.gif" alt="drawing" width="500" />
  <img src="/doc/test_larger.gif" alt="drawing" width="500" /> 
</p>

## Dependency
- Python 3
- [PyTorch](https://pytorch.org/)
- [PyTorch Geometric](https://pytorch-geometric.readthedocs.io/en/latest/#)
- [gtsam](https://gtsam.org/) (Georgia Tech Smoothing and Mapping library)
  ```
  git clone -b emex --single-branch https://bitbucket.com/jinkunw/gtsam
  cd gtsam
  mkdir build && cd build
  cmake ..
  sudo make install
  ```
- [pybind11](https://github.com/pybind/pybind11) (pybind11 — Seamless operability between C++11 and Python)
  ```
  git clone https://github.com/pybind/pybind11.git
  cd pybind11
  mkdir build && cd build
  cmake ..
  sudo make install
  ```
 
## Compile
You can use the following commands to download and compile the package.
```
git clone https://github.com/RobustFieldAutonomyLab/DRL_Graph_exploration.git
cd DRL_graph_exploration
mkdir build && cd build
cmake ..
make
```

## Issues
There is an unsolved memory leak issue in the C++ code. So we use the python subprocess module to run the simulation training. The data in the process will be saved and reloaded every 10000 iterations.

## How to Run?
- To run saved policy:
    ```
    cd DRL_graph_exploration/scripts
    python3 test.py
    ```
- To train your own policy:
    ```
    cd DRL_graph_exploration/scripts
    python3 train_terminal_run.py
    ```
 
- To show average reward during the training:
    ```
    cd DRL_graph_exploration/data
    tensorboard --logdir=torch_logs
    ```

## Cite

Please cite [our paper]() if you use any of this code: 
```
@inproceedings{ExplorGraphDRL2020,
  title={Autonomous Exploration Under Uncertainty via Deep Reinforcement Learning on Graphs},
  author={Chen, Fanfei and Martin, John D. and Huang, Yewei and Wang, Jinkun and Englot, Brendan},
  booktitle={},
  year={2020},
}
```

## Reference
- [em_exploration](https://github.com/RobustFieldAutonomyLab/em_exploration)