# 10703-hw4

# Setting up the environment
Before you begin, you need to install the required dependencies. If you work locally or on AWS, We encourage the use of a virtual environment in order to help you manage your dependencies correctly. Checkout official documentation for VirtualEnv or Conda for this. <br > 
If you already have a virtual environment setup from previous homeworks, you may continue to use them. Otherwise, you can create a new environment and start installing the dependencies. <br > 
To help you with this process, we have provided sample commands in Conda to install each dependency. <br >
First, you need to install the following packages: <br > 
numpy <br > 
gym <br > 
ruamel.yaml <br> 
tensorflow 2.x with GPU support (NOTE: if you use the conda install tensorflow command, conda will install CUDA 10.1 along with tensorflow 2.2.0 in your virtual environment. This setup is fine and tested, and the CUDA 10.1 will not interfere with any existing CUDA installation on your system) <br> 
```
conda install numpy
conda install -c conda-forge gym
conda install tensorflow
conda install ruamel.yaml
```
To gain access to the InvertedPendulum environment, you need to install it via pip.  <br > 
Inside a virtual environment with pip installed (this can be a conda environment or venv environment), run <br >
```
$ pip3 install pybullet --upgrade --user
```
After you finish the previous command, try running the following demo: <br > 
```
 python3 -m pybullet_envs.examples.enjoy_TF_AntBulletEnv_v0_2017may
```
You should see a window pop-up with a quadruped robot walking in a grid-world. If you do not observe this, your installation is incorrect and you need to troubleshoot/reinstall your pybullet environment. <br > 

# Directory Structure and Usage
Your starter code should have the following structure: <br >
```
-hw4-handout 
 -src 
  fake_env.py (helper code for sampling model trajectorys, DO NOT EDIT.) 
  __init__.py (empty script to make src a Python module. No need to edit.) 
  mbpo.py (contains the MBPO class, which you need to implement.) 
  pe_model.py (contains the Probabilistic Ensemble of Neural Networks used as our dynamics model. You need to implement part of this file.) 
  td3.py (contains the Time-Delayed Deep Deterministic policy gradient (TD3) algorithm. You need to implement part of this file.) 
  utils.py (contains the custom Replay Buffer. You are strongly advised to read over its documentation, but you don't need to edit this file.) 
 -scripts 
  runner.py (a script that loads the yaml config file and runs your MBPO class' training routine. No need to edit.) 
 -configs 
  InvertedPendulum.yml (a yaml config file that contains all the hyperparameters for your implementation of TD3 and MBPO.) 
 -results 
 run.sh (a shell script to add the current directory to Python path and run the runner script. No need to edit.) 
 README.md 
```
To run your MBPO code: 
```
$ bash run.sh
```
To run your TD3 code without MBPO:
Edit InvertedPendulum.yml and set enable_MBPO to false. Then
```
$ bash run.sh
```
To run a simple sanity check on your PE model implementation: 
In src/, run
```
$ python3 pe_model.py 
```
The graphs are generated automatically and they will be in the results/ folder. Include the graphs in the writeup. 


# Running time
For the TD3 training, our reference implementation converges in 55K-60K timesteps. Training time for 75K timesteps is about 15 minutes, on a Desktop with 3900X and RTX 2080. <br > 
For the MBPO training, our reference implementation converges in 20K real environment timesteps. Training time for 20K timesteps is about 60 minutes, on a Desktop with 3900X and RTX 2080. <br > 
Both runtimes above assume the default hyperparameters in configs/InvertedPendulum.yml. <br >
For the sanity check on PE model, our reference implementation converges in 70-150 epochs. Training time is <5 minutes.

# References 
@article{janner2019mbpo,
  author = {Michael Janner and Justin Fu and Marvin Zhang and Sergey Levine},
  title = {When to Trust Your Model: Model-Based Policy Optimization},
  journal = {arXiv preprint arXiv:1906.08253},
  year = {2019}
}
@inproceedings{fujimoto2018addressing,
  title={Addressing Function Approximation Error in Actor-Critic Methods},
  author={Fujimoto, Scott and Hoof, Herke and Meger, David},
  booktitle={International Conference on Machine Learning},
  pages={1582--1591},
  year={2018}
}
@MISC{coumans2019,
author =   {Erwin Coumans and Yunfei Bai},
title =    {PyBullet, a Python module for physics simulation for games, robotics and machine learning},
howpublished = {\url{http://pybullet.org}},
year = {2016--2019}
}
