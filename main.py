import gymnasium as gym
import grid
from stable_baselines3 import DQN, A2C
import os


logdir = "grid/log"

if not os.path.exists(logdir):
        os.makedirs(logdir)

h = [
    [0, 0, 0, 0, 0, 1, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 1, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 1, 1, 1],
    [0, 0, 0, 0, 0, 0, 0, 0, 1, 1],
    [1, 0, 0, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 0, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 0, 0, 0, 0, 0, 0],
    [1, 1, 1, 1, 1, 1, 0, 0, 0, 0]
    ]

options = {
        'start': 0,
        'goal': 99
    }

env = gym.make('grid/GridWorld-v0', map=h, options=options)
env.reset()

#model = DQN("MlpPolicy", env, verbose=1, tensorboard_log=logdir)
model2 = A2C("MlpPolicy", env, verbose=1, tensorboard_log=logdir)


TIMESTEP = 100000
model2.learn(total_timesteps = TIMESTEP, reset_num_timesteps=False, tb_log_name="TEST_nw2-A2C")


env.close()
