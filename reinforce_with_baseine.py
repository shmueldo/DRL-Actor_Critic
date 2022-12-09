import gym
import numpy as np
import tensorflow.compat.v1 as tf
import collections
from ModifiedTensorBoard import *
import os
from datetime import datetime
import time


# optimized for Tf2
tf.disable_v2_behavior()

SEED = 42

env = gym.make('CartPole-v1')
np.random.seed(SEED)
env.seed(SEED)
tf.set_random_seed(SEED)

algorithm_name = "Reinforce wt baseline"

class PolicyNetwork:
    def __init__(self, state_size, action_size, learning_rate, name='policy_network'):
        self.state_size = state_size
        self.action_size = action_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.action = tf.placeholder(tf.int32, [self.action_size], name="action")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 12], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [12], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [12, self.action_size], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [self.action_size], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)

            # Softmax probability distribution over actions
            self.actions_distribution = tf.squeeze(tf.nn.softmax(self.output))
            # Loss with negative log probability
            self.neg_log_prob = tf.nn.softmax_cross_entropy_with_logits_v2(logits=self.output, labels=self.action)
            self.loss = tf.reduce_mean(self.neg_log_prob * self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)

class ValueNetwork:
    def __init__(self, state_size, learning_rate, name='state_value_network'):
        self.state_size = state_size
        self.learning_rate = learning_rate

        with tf.variable_scope(name):

            self.state = tf.placeholder(tf.float32, [None, self.state_size], name="state")
            self.R_t = tf.placeholder(tf.float32, name="total_rewards")

            tf2_initializer = tf.keras.initializers.glorot_normal(seed=0)
            self.W1 = tf.get_variable("W1", [self.state_size, 8], initializer=tf2_initializer)
            self.b1 = tf.get_variable("b1", [8], initializer=tf2_initializer)
            self.W2 = tf.get_variable("W2", [8, 1], initializer=tf2_initializer)
            self.b2 = tf.get_variable("b2", [1], initializer=tf2_initializer)

            self.Z1 = tf.add(tf.matmul(self.state, self.W1), self.b1)
            self.A1 = tf.nn.relu(self.Z1)
            self.output = tf.add(tf.matmul(self.A1, self.W2), self.b2)
            
            # Calculate the advantage function
            self.advantage = self.R_t - tf.squeeze(self.output)
            
            # Mean squared error loss
            self.loss = tf.losses.mean_squared_error(predictions=self.output, labels=self.R_t)
            self.optimizer = tf.train.AdamOptimizer(learning_rate=self.learning_rate).minimize(self.loss)


def run(discount_factor, policy_learning_rate ,sv_learning_rate):
    rewards, mean_rewards, losses = [], [], []
    # Define hyperparameters
    state_size = 4
    action_size = env.action_space.n

    max_episodes = 5000
    max_steps = 501
    # discount_factor = 0.99
    # policy_learning_rate = 0.0004
    # sv_learning_rate = 0.0004
    discount_factor = discount_factor
    policy_learning_rate = policy_learning_rate
    sv_learning_rate = sv_learning_rate
    render = False

    # Initialize the policy and the state-value network
    tf.reset_default_graph()
    policy = PolicyNetwork(state_size, action_size, policy_learning_rate)
    state_value = ValueNetwork(state_size, sv_learning_rate)

    # Start training the agent with REINFORCE algorithm
    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        solved = False
        Transition = collections.namedtuple("Transition", ["state", "action", "reward", "next_state", "done"])
        episode_rewards = np.zeros(max_episodes)
        average_rewards = 0.0

        for episode in range(max_episodes):
            state = env.reset()
            state = state.reshape([1, state_size])
            episode_transitions = []

            for step in range(max_steps):
                actions_distribution = sess.run(policy.actions_distribution, {policy.state: state})
                action = np.random.choice(np.arange(len(actions_distribution)), p=actions_distribution)
                next_state, reward, done, _ = env.step(action)
                next_state = next_state.reshape([1, state_size])

                if render:
                    env.render()

                action_one_hot = np.zeros(action_size)
                action_one_hot[action] = 1
                episode_transitions.append(Transition(state=state, action=action_one_hot, reward=reward, next_state=next_state, done=done))
                episode_rewards[episode] += reward

                if done:
                    if episode > 98:
                        # Check if solved
                        average_rewards = np.mean(episode_rewards[(episode - 99):episode+1])
                    
                    print("Episode {} Reward: {} Average over 100 episodes: {}".format(episode, episode_rewards[episode], round(average_rewards, 2)))
                    if average_rewards > 475:
                        print(' Solved at episode: ' + str(episode))
                        solved = True
                    break
                state = next_state

            if solved:
                break

            # Compute Rt for each time-step t and update the network's weights
            for t, transition in enumerate(episode_transitions):
                # Compute Gt
                total_return = sum(discount_factor ** i * t.reward for i, t in enumerate(episode_transitions[t:])) # Rt
                
                # Compute advantage
                feed_dict = {state_value.state: transition.state, state_value.R_t: total_return}
                advantage = sess.run(state_value.advantage, feed_dict)
                
                # Update the state_value network weights
                feed_dict = {state_value.state: transition.state, state_value.R_t: total_return}
                _, loss_state = sess.run([state_value.optimizer, state_value.loss], feed_dict)

                # Update the policy network weights
                feed_dict = {policy.state: transition.state, policy.R_t: advantage, policy.action: transition.action}
                _, loss_policy = sess.run([policy.optimizer, policy.loss], feed_dict)
            
            
            rewards.append(episode_rewards[episode])
            mean_rewards.append(average_rewards)
            losses.append(loss_policy)
    return episode, rewards, mean_rewards, losses


if __name__ == '__main__':
    for lr in [0.0001, 0.0002, 0.0005, 0.001, 0.01]:
        last_episode, rewards, mean_rewards, losses = run(discount_factor=0.99, policy_learning_rate=0.0004, sv_learning_rate=lr)
        with open('test_lr_{}.npy'.format(lr), 'wb') as f:
            np.save(f, last_episode)
            np.save(f, rewards)
            np.save(f, mean_rewards)
            np.save(f, losses)