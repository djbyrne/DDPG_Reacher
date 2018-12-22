[//]: # (Image References)

[image1]: https://user-images.githubusercontent.com/10624937/43851024-320ba930-9aff-11e8-8493-ee547c6af349.gif "Trained Agent"
[image2]: https://user-images.githubusercontent.com/10624937/43851646-d899bf20-9b00-11e8-858c-29b5c2c94ccc.png "Crawler"


# DDPG Reacher

### Introduction

This experiment implements the DDPG algorithm to train a mechanical arm to reach for a moving target inside the unity ML-Agents virtual [Reacher](https://github.com/Unity-Technologies/ml-agents/blob/master/docs/Learning-Environment-Examples.md#reacher) environment. In this environment, a double-jointed arm can move to target locations.

![Trained Agent][image1]

### Rewards

A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

### State Space

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 

### Action Space
Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

### Solving the Environment
The environment is considered solved when the average score of all agents in the environment (in this case 20) for a period of 100 episodes is 30 or above.

### Setup

1. Create (and activate) a new environment with Python 3.6.

	- __Linux__ or __Mac__: 
	```bash
	conda create --name drlnd python=3.6
	source activate drlnd
	```
	- __Windows__: 
	```bash
	conda create --name drlnd python=3.6 
	activate drlnd
	```

2. Clone the repository and install dependencies.
```bash
git clone https://github.com/djbyrne/DDPG_Reacher.git
cd ddpg_reacher
pip install .
```
3.Download the environment from one of the links below.  You need only select the environment that matches your operating system:

- **_Twenty (20) Agents_**
	- Linux: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Linux.zip)
	- Mac OSX: [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher.app.zip)
	- Windows (32-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86.zip)
	- Windows (64-bit): [click here](https://s3-us-west-1.amazonaws.com/udacity-drlnd/P2/Reacher/Reacher_Windows_x86_64.zip)

(_For Windows users_) Check out [this link](https://support.microsoft.com/en-us/help/827218/how-to-determine-whether-a-computer-is-running-a-32-bit-version-or-64) if you need help with determining if your computer is running a 32-bit version or 64-bit version of the Windows operating system.

4. Place the file in the project directory and change the environment path in the notebook DDPG_Reacher.ipynb

### Instructions

All code for this project is contained in the DDPG_Reacher.ipynb notebook. As such you just need to run the cells in order to see the results.
