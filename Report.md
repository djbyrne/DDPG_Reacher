# Intro

This project looks at implementing the Deep Deterministic Policy Gradient (DDPG) algorithm in order to solve Reacher environment.

This implementation used the ddpg pendulum example from the Udacity deep reinforcment learning repository as a foundation and was adapted to suite the Reacher environment. This project can be found [here](https://github.com/udacity/deep-reinforcement-learning/tree/master/ddpg-pendulum)

The Reacher environment provided an interesting challenge and required several subtle but integral changes to the my initial code in order for the agent to train successfully.

# The environment

![Trained Agent][image1]

The environment used for this project was built using the Unity [ml-agents](https://github.com/Unity-Technologies/ml-agents) framework.

In this environment, a double-jointed arm can move to target locations. A reward of +0.1 is provided for each step that the agent's hand is in the goal location. Thus, the goal of your agent is to maintain its position at the target location for as many time steps as possible.

The observation space consists of 33 variables corresponding to position, rotation, velocity, and angular velocities of the arm. 

Each action is a vector with four numbers, corresponding to torque applicable to two joints. Every entry in the action vector should be a number between -1 and 1.

# DDPG
DQN introduces a lot of changes to the traditional Q learning algorithm and isn't just replacing the Q table with a neural network.
As described in Deep Minds groundbreaking paper [Playing Atari with Deep Reinforcement Learning](https://arxiv.org/pdf/1312.5602.pdf) there are 
several advancements that make this algorithm work. Below I will go through the core techniques used to successfully implement the DQN algorithm.

## Models

Like I mentioned previously, one of the big changes in how we approach Q learning is the introduction of using a neural network to replace the Q table. Instead of storing every Q value in a look up table corresponding to its state and action, we can use the neural network to approximate the Q value of any state/action pair. This allows us to tackle infinitely more complex tasks that were out of reach for the simple tabular approach.

This base implementation of this project uses a simple 2 layer multi layer perceptron(MLP). The model takes in the 37 state features as input and returns the best Q values for each of our 4 possible actions given the current state as output. We then take the max of these values as our best action. For the base model of this project I kept the same architecture and parameters as the lunar lander excerise in order to get a solid working agent to begin. The base model looks like the following:

| Layers           |Parameters           |
|:-------------:| :-------------:| 
| Dense Layer| 16| 
| ReLU activation| NA|   
| Dense Layer| 32| 
| ReLU activation| NA| 
|Dense Layer|4| 

## Memory

One of the main improvements provided by the DQN paper is the use of Experience Replay. This acts like episodic memory and allows our agent to continually learn from its previous experiences as opposed to throughing them away at the end of each step. The use of experience replay has a few steps:

1) We need to observe our environment for several iterations in order to fill up our memory buffer with enough experiences to learn from. At each time step we will add our experience to our memory buffer. This experience stores the state, action, reward, next state and done variables for that step.

2) When we have enough memories stored (at least equal to our batch size) we can start learning. At each time step, after we add our experience to our buffer we check and see if we want to perform a learning step (every 4 steps) if so we sample from our memory buffer

3)Once we have this sample of previous experiences we can train on them. This will allow our agent to learn how to correctly approxminate the Q values. The details of this method will be discussed in the next section

## Agent

Next we have our agent. This of course the main portion of the project and ties everything together. The details of the agent are in the notebook itself but I will give a breif overview of the key parts

The agent can be intialised with several augmentations. The base model is set by default. This used only the standard DQN moel. Double Learning, Duelling Networks and Prioritized Experience Replay can all be added and removed during initialization.

The step function in the agent is taken after the environment step function and shouldn't be confused. Here we take in the state, action, reward, next_state and done variables from the last step and add it to our experience replay buffer. Then the agent carries out the learn function.

Inside the learn function we take a sample from our experience replay buffer and iterate through that sample of experience. For each of these we update the Q value corresponding to the state action pair. This is done by computing the loss between the target and the expected prediction. In the base model the target is the predicted Q value of the next states and choosing the best best action using our target model. We then calculate the discounted rewards of that target model prediction to form our new target. This is then compared to the local models prediction of the Q value given the initial state. After this we carry out a soft update of the target model.

The last function worth discussing is the act method. Here we take in the current state of the environment and get the best prediction from our local model. Next we use the epsilon greedy strategy to determine wether we use our models action or if we use a random action. 

# Training

The training portion of the notebook contains the main game loop iterating through the environment and utilising our agent. As with most machine learning problems, a lot of the improvements come from hyper parameter tuning. This can often taken longer than building the actual algorithm itself. Unfortunately I couldn't dedicate too much time to hyper parameter tuning and was only able to test a few changes for each type of model. Below are the parameters that I experimented with. Each parameter change was added in and tested individually in order to identify which parameter changes gave the best results.

| Parameter | Params 1 |  Params 2  |
|:-------------:| :-------------:| :-------------:| 
|Layer 1    | 16     |   32     |
|Layer 2    | 32     |   64     |
|Learning Rate    | 0.0005     |   0.0001     |
|Batch Size    | 64     |   32     |
|Buffer Size    | 100000     |  200000     |


The experiments showed that the hyperparameters didnt make a huge improvement, with most models reaching a stable score of 13 within 800 episodes and a score ~15-17 after 2000. The variation on the agents final score was mostly influenced by the type of model used. The hyperparameters mainly effected how quickly the agent could reach a score of 13+. For example, the base DQN with the initial set of parameters learned to get a score of 13+ within ~450 episodes where as the same model with the second set of parameters took ~700 episodes. Both of these models achieved a final average score of 16 after 2000 episodes and could achieve the max score of 25. I believe that the model is capable achieving a better highscore with correct hyperparameter tuning and more training episodes. Below is the training graph of the base DQN agent running with the 2nd set of hyperparameters.

![DQN_2000_Episodes](/images/dqn_2000.png)


# Results

|Model| Episodes to reach 13+|
|:-------------:| :-------------:|
|DQN|430|
|Double DQN| 458|
|Duelling DQN| 473|
|Double Duelling DQN| 509|
|PER DQN| 898 |

I conclusion, all implementations of the DQN algorithm were capable of beating the environment (score of 13+) in under 550 episodes). My experiments have shown that the addition of the duelling network and double learning provided the best results for the navigation environment given 2000 training episodes. The agent was able to attain a high score of 17.43 on average over 100 episodes and frequently hit the high score of 25. I believe this is due to the duelling networks ability to generalise better than the standard DQN. On top of this, the addition of double learning made the agent more robust and stable, as you can see from the graph below. 

![DDDQN](/images/dueling_double_17.png)

However it is worth pointing out that these results were only a little better than the standard agent, which could reach a score of ~16. 

# Future Work

## Furthur Hyperparameter Tuning
Like most machine learning problems I believe that this agent could be improved to reach a much higher average through hyperparameter tuning. With my limited amount of time to spend of the project I was only able to test 2 variations for each parameter. With more tuning I believe the agent could perform significantly better. Potentially I would like to use something like AuotMl or genetic algorithms to learn the best hyperparameters for this environment.

## Testing More Complicated Environments
As well as this I think that the reason that I did not see a big improvement from the additions is down to the fact that the environment itself is quite simple. Due the this, it is possible that the benefits of these improvements are not being seen. I would like to try these methods on a more complicated environment in the future such as the atari games.


