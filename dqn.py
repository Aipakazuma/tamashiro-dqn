# -*- coding: utf-8 -*-

import tensorflow as tf
import gym 


ENV_NAME = 'Breakout-v0'
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool("train", False,
                         """Start DQN training.""")
tf.app.flags.DEFINE_bool("test", False,
                         """Start DQN test.""")
tf.app.flags.DEFINE_integer("max_steps", 10000,
                            """Execute max step number.""")


def define_model():
    """Modelの設定."""
    pass


def optimizer():
    """Optimizerの設定."""
    pass


def loss():
    """loss関数の設定."""
    pass


def define_env():
    """gym環境を取得"""
    env = gym.make(ENV_NAME)
    return env


def train():
    """Model training."""
    env = define_env()
    # episode 
    for n_episode in range(10):
        observation = env.reset()
        print(observation)
        # step
        for t in range(1000):
            # observationを画面へ表示
            env.render()
            # actionを適当に決める
            action = env.action_space.sample()
            # actionを渡してstepし、次のobservation(s`)や報酬を受け取る
            observation, reward, done, info = env.step(action)
            # gameが終了したらbreakする
            if done:
                print('Episode finished after %d timesteps' % (t + 1))
                break

    env.close()


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
