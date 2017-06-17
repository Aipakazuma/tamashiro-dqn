# -*- coding: utf-8 -*-

import tensorflow as tf
import gym 
import numpy as np
from collections import namedtuple, deque
import random


# openai gym
ENV_NAME = 'Breakout-v0'
np.seed(123)

# deep q network
# 学習前に事前に確保するexperience replay
INITIAL_REPLAY_SIZE = 2000

# tensorflow command argument
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool("train", False,
                         """Start DQN training.""")
tf.app.flags.DEFINE_bool("test", False,
                         """Start DQN test.""")
tf.app.flags.DEFINE_integer("max_steps", 10000,
                            """Execute max step number.""")


class Agent():
    """DQNのAgent."""

    def __init__(self, env, batch_size):
        """初期化."""
        # experice memory
        # keras rlを参考
        self.experience = namedtuple('Experience',
                                     'state0, action, reward, state1, terminal1')
        self.experience_memory = deque()
        self.env = env
        self.batch_size = batch_size


    def sampling(self):
        return random.sample(self.experience_memory, self.batch_size)


    def fit(self):
        # episode
        action_steps = 0
        for n_episode in range(10):
            observation = self.env.reset()
            # step
            for t in range(1000):
                # observationを画面へ表示
                self.env.render()
                # actionを適当に決める
                action = self.env.action_space.sample()
                # actionを渡してstepし、次のobservation(s`)や報酬を受け取る
                state0 = observation.copy()
                observation, reward, done, info = self.env.step(action)
                self.experience_memory.append(self.experience(state0=state0, action=action,
                                                              reward=reward, state1=observation, 
                                                              terminal1=done))
                # if LIMIT_EXPERIENCE < len(self.experience_memory):
                #     self.experience_memory.popleft()

                # action stepsがmemoryサイズを超えないと学習させない
                # memoryサイズがある程度ないとmini batchが作れないため
                if action_steps < INITIAL_REPLAY_SIZE:
                    pass

                action_steps += 1

                # gameが終了したらbreakする
                if done:
                    print('Episode finished after %d timesteps' % (t + 1))
                    break

        self.env.close()


def define_model():
    """Modelの設定."""
    pass


def optimizer():
    """Optimizerの設定."""
    pass


def loss():
    """loss関数の設定."""
    pass


def train():
    """Model training."""
    env = gym.make(ENV_NAME)
    env.seed(123)
    agent = Agent(env=env, batch_size=32)
    agent.fit()


def test():
    """Model test."""
    pass


def main(argv=None):
    """Tensorflowのmain関数."""
    if FLAGS.train:
        train()

    if FLAGS.test:
        test()


if __name__ == '__main__':
    tf.app.run()
