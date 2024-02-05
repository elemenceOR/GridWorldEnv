from __future__ import annotations
import numpy as np
from gymnasium import Env, spaces


class GridWorldEnv(Env):
    PATH = 0
    ACTIONS = {
        0: (-1, 0),  # UP
        1: (1, 0),  # DOWN
        2: (0, -1),  # LEFT
        3: (0, 1)  # RIGHT
    }

    def __init__(self, map, options=None):
        self.mymap = self.create_map(map)
        self.row_num, self.col_num = self.mymap.shape

        self.start = self.get_start_and_goal('start', options)
        self.goal = self.get_start_and_goal('goal', options)

        oS = self.row_num * self.col_num # obbservation space
        aS = 4 # action space

        self.action_space = spaces.Discrete(aS)
        self.observation_space = spaces.Discrete(oS)

    def reset(self, seed=None, options=None):
        self.agent_pos = self.start
        self.reward = self.get_reward(*self.agent_pos)
        self.done = self.reach_goal()

        return self.get_obs(), self.get_info()

    def step(self, action):
        row, col = self.agent_pos
        xx, yy = self.ACTIONS[action]

        # calculating the next state coordinates
        next_row = row + xx
        next_col = col + yy

        self.reward = self.get_reward(next_row, next_col)

        # check whther the calculated next state is within the defined grid before taking the step
        if self.in_bound(next_row, next_col) and self.is_free(next_row, next_col):
            self.agent_pos = (next_row, next_col)
            self.done = self.reach_goal()
            #print("Done")
        else:
            self.done = True
            #print("terminated")
            self.reward = self.get_reward(next_row, next_col)
            
        return self.get_obs(), self.reward, self.done, False, self.get_info()
    
    def close(self):
        pass

    # -- utility functions --

    # to convet the input map to a 2d array
    def create_map(self, map):
        if isinstance(map, list):
            map_str = np.asarray(map, dtype='c')
            map_int = np.asarray(map_str, dtype=int)
            return map_int

    # reward function
    def get_reward(self, row, col):
        if not self.in_bound(row, col):
            return -10
        elif not self.is_free(row, col):
            return -10
        elif (row, col) == self.goal:
            return 100
        else:
            return 0
        #if (row, col) == self.goal:
        #    return 1
        #else:
        #    return 0

    # check wether the agent is within the defined grid
    def in_bound(self, row, col):
        return 0 <= row < self.row_num and 0 <= col < self.col_num

    # check the given coordinate is a free space
    def is_free(self, row, col):
        return self.mymap[row, col] == self.PATH

    def reach_goal(self):
        return self.agent_pos == self.goal

    # convert rows and columns to states
    def to_state(self, row, col):
        return row * self.col_num + col

    # convert the agent positions into a state and use it as an observation
    def get_obs(self):
        return self.to_state(*self.agent_pos)

    # return the agent position as a dictionary. (as requires by gym)
    def get_info(self):
        return {'agent pos': self.agent_pos}

    # parse the user define starting and goal coordinates
    def get_start_and_goal(self, state_name, options):
        state = options[state_name]

        if isinstance(state, int):
            return self.to_xy(state)
        elif isinstance(state, tuple):
            return state
        else:
            raise TypeError(
                f'Allowed types for `{state_name}` are int or tuple.')

    # convet the given start and end positions to x,y coordinates
    def to_xy(self, s):
        return (s // self.row_num, s % self.col_num)

