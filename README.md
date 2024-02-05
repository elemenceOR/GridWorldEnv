# GridWorldEnv
The use of prebuild environments are sufficient to understand how each RL algorithm works
and obtain initial results, but to test out the potential of each RL algorithm and compare them
properly, a custom environment is needed. A custom environment gives few advantages such
as the ability to define custom reward functions, and ability to decide what are the outcomes
for the actions taken by the RL agent. For example, the “Frozen Lake” environment is built to
process only a binary sparse reward system where the agent will get a reward only at the end
of a successful episode. But it is requiring to understand the effects of other types of reward
systems. This allows to choose the most effective reward system for a given algorithm.

## Installation and Using
Before using the environment, install it using
```bash
pip install -e grid
```

* Use main.py to test out A2C and DQN
* Use qL_agent_cstm.py to test the custom q learning algorythm

> [!TIP]
> The grid can be customized to any size as long as it a squre. ( Eg: 5x5, 10x10, 20x20 )

> [!TIP]
> Starting and Ending can be anything as long as there is no obstacle. 


## Basic Environment Structure
![Screenshot 2024-02-05 150855](https://github.com/elemenceOR/GridWorldEnv/assets/52843991/8205e3ab-6966-4c46-b65a-ded551f0b5a3)

## Reward function
The agent has three termination conditions and rewards are allocated accordingly.

* Agent reaches the goal
* Agent reaches an obstacle
* Agent tries to go out of the grid

It is possible to change it to a binary parsed reward system where agent get a reward only at the goal position.

Also possible to give the agent a relatively small penalty for each step it takes,
so that the agent leans to take the least number of steps to reach the goal.

**Shaped Reward system:**
```python
def get_reward(self, row, col):
        if not self.in_bound(row, col):
            return -10
        elif not self.is_free(row, col):
            return -10
        elif (row, col) == self.goal:
            return 100
        else:
            return 0
```

**Binary Sparced Reward System:**
```python
    def get_reward(self, row, col):
        if (row, col) == self.goal:
            return 1
        else:
            return 0
```

**With Time Penalty:**
```python
def get_reward(self, row, col):
        if not self.in_bound(row, col):
            return -10
        elif not self.is_free(row, col):
            return -10
        elif (row, col) == self.goal:
            return 100
        else:
            return -0.01
```


## Results

**With Custom Q Learning Agent:**

_Binary Sparce Rewards:_

![Picture2](https://github.com/elemenceOR/GridWorldEnv/assets/52843991/fef83d71-0fc3-44eb-9cef-e06393dc1370)

_Shaped Rewards:_

![Picture4](https://github.com/elemenceOR/GridWorldEnv/assets/52843991/87149290-0a84-4806-8588-697bd23933ba)


**With DQN adn A2C from StableBaseline3:**

_Binary Sparce Rewards:_

![dqnvsa2c](https://github.com/elemenceOR/GridWorldEnv/assets/52843991/e8ff5833-e8d5-4290-a99e-88efb82d31f3)

_Shaped Rewards:_
* Agent reaches the goal – reward +1
* Agent reaches an obstacle - reward -0.01
* Agent tries to go out of the grid – reward -0.1
  
![dqna2c4](https://github.com/elemenceOR/GridWorldEnv/assets/52843991/d5d103db-c104-47da-b60f-7751431b20d0)

_Shaped Rewards with Time Penalty:_
* Time penalty - reward -0.01

![63compsvg](https://github.com/elemenceOR/GridWorldEnv/assets/52843991/0a81eca2-a26e-4d00-a03d-f80a922c72a3)

> [!CAUTION]
> Time penalty does not work on custom Q learning. Still working on it. 

## Summery
In summary, the comparison of policy-based and non-policy-based Reinforcement Learning (RL) algorithms in a 10x10 grid world environment revealed distinct performance differences. Tabular Q learning, a non-policy-based algorithm, demonstrated faster convergence compared to DQN due to the environment's simplicity and the efficiency of updating a smaller state table. DQN, on the other hand, faced challenges with sample efficiency in the small environment, making it less suitable. A2C, a policy-based algorithm, outperformed DQN with a higher convergence rate attributed to its entropy regularization exploration strategy, enhancing sample efficiency. Despite tabular Q learning showing better performance in the given environment, A2C exhibited quicker convergence in terms of iterations due to its sample efficiency. Ultimately, for the specified grid world environment, tabular Q learning proved to be more efficient than both DQN and A2C.
