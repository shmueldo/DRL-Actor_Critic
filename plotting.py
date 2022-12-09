import gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
from ModifiedTensorBoard import *
import os
from datetime import datetime
import time

if __name__ == '__main__':
    for lr in [0.0001, 0.0002, 0.0005, 0.001, 0.01]:
        with open('test_lr_{}.npy'.format(lr), 'rb') as f:
            last_episode = np.load(f)
            rewards = np.load(f)
            mean_rewards = np.load(f)
            losses = np.load(f)
        
        algorithm_name = "baseline_lr={}".format(lr)
        # Custom tensorboard object
        tensorboard = ModifiedTensorBoard(algorithm_name,
                                        log_dir="{}logs/{}-{}".format(os.getcwd() +r"/",
                                        algorithm_name, datetime.now().strftime("%d%m-%H%M")))
        for episode in range(last_episode):
            #update tensor board step
            tensorboard.step = episode
            tensorboard.update_stats(episode_rewards = int(rewards[episode]), mean_reward= mean_rewards[episode], loss=losses[episode])