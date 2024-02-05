import gymnasium as gym
import numpy as np
import matplotlib.pyplot as plt
import pickle
import grid



fig = plt.figure()
ax1 = fig.add_subplot(1,1,1)


def run(episodes, is_training=True):

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
    if(is_training):
        q = np.zeros((env.observation_space.n, env.action_space.n))
    else:
        f = open('qn/frozen_lake10x10.pkl', 'rb')
        q = pickle.load(f)
        f.close()

    learning_rate_a = 0.9 # alpha or learning rate
    discount_factor_g = 0.9 # gamma or discount factor.

    epsilon = 1  # 1 = 100% random actions
    epsilon_decay_rate = 0.0001  # epsilon decay rate. 1/0.0001 = 10,000

    actions = [0, 1, 2, 3]

    rewards_per_episode = np.zeros(episodes)

    for i in range(episodes):
        state, _ = env.reset()

        terminated = False

        max_q_val = 0   

        while(not terminated):
            if is_training: 
                if np.random.uniform(0,1) <= epsilon:
                    action = np.random.choice(actions)  # actions: 0=up,1=down,2=left,3=right
                else:
                    for a in actions:
                        q_val = q[state][a]
                        if q_val >= max_q_val:
                            action = a
                            max_q_val = q_val

            new_state,reward,terminated,_,info = env.step(action)

            if is_training:
                q[state,action] = q[state,action] + learning_rate_a * (
                    reward + discount_factor_g * np.max(q[new_state,:]) - q[state,action]
                )
                
            state = new_state

        epsilon = max(epsilon - epsilon_decay_rate, 0)

        if(epsilon==0):
            learning_rate_a = 0.0001
        
        if reward == 1:
            rewards_per_episode[i] = 1

        if i % 1000 == 0:
            print(i)   
    
    
    print(rewards_per_episode)

    env.close()

    sum_rewards = np.zeros(episodes)
    for t in range(episodes):
        sum_rewards[t] = np.sum(rewards_per_episode[max(0, t-100):(t+1)])
    
    # normalize the result between 0 and 1
    normalized_y = (sum_rewards - np.min(sum_rewards)) / (np.max(sum_rewards) - np.min(sum_rewards))
    print(normalized_y)

    # calculate moving average
    ma = np.convolve(normalized_y, np.ones(100), mode="valid") / 100


    plt.figure(figsize=(6, 4), dpi=300)
    plt.title("Cumilative rewards per 100 episodes")
    plt.xlabel("Episode")
    plt.ylabel("Reward")
    plt.plot(normalized_y, linewidth=1, alpha=0.1, label="Cumilative rewards for 100 epsiodes")
    plt.plot(ma, linewidth=0.65, label="Moving average(100)")
    plt.legend(loc='upper left', fontsize=8)
    plt.savefig('qn/frozen_lake10x10.svg')

    
    if is_training:
        f = open("qn/frozen_lake10x10.pkl","wb")
        pickle.dump(q, f)
        print(q)
        f.close()

if __name__ == '__main__':
    run(15000, is_training=True)
