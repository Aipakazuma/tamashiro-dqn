# -*- coding: utf-8 -*-

import tensorflow as tf
import gym 
import numpy as np
from collections import namedtuple, deque
import random
import cv2


# openai gym
ENV_NAME = 'Breakout-v0'
np.random.seed(123)

# deep q network
# 学習前に事前に確保するexperience replay
INITIAL_REPLAY_SIZE = 2000
TRAIN_INTERVAL = 200
TARGET_UPDATE_INTERVAL = 500
WINDOW_LENGTH = 4
LEARNING_RATE = 0.00025  # RMSPropで使われる学習率
MOMENTUM = 0.95  # RMSPropで使われるモメンタム
MIN_GRAD = 0.01  # RMSPropで使われる0で割るのを防ぐための値
LIMIT_EXPERIENCE = 400000  # Reply memoryの上限
GAMMA = 0.99  # 割引率

# tensorflow command argument
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_bool("train", False,
                         """Start DQN training.""")
tf.app.flags.DEFINE_bool("test", False,
                         """Start DQN test.""")
tf.app.flags.DEFINE_integer("max_episode", 10,
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
        self.update_target_network = None


    def inference(self, x, inference_name):
        """推論処理.

        ニューラルネットワークのレイヤーで構成されるモデルのこと.
        """
        weights_list = []
        def _get_weights(name, shape, stddev=1.0):
            return tf.get_variable(inference_name + '_weights_' + name,
                                   shape,
                                   initializer=tf.truncated_normal_initializer(stddev=stddev))

        def _get_biases(name, shape, value=0.0):
            return tf.get_variable(inference_name + '_biases_' + name,
                                   shape,
                                   initializer=tf.constant_initializer(value))


        with tf.name_scope('flatten'):
            flatten = tf.reshape(x, shape=[-1, 100800])

        with tf.name_scope('fc1'):
            weights = _get_weights(name='fc1', shape=[100800, 32])
            biases = _get_biases(name='fc1', shape=[32])
            fc1 = tf.nn.relu(tf.matmul(flatten, weights) + biases)
            weights_list.append(weights)
            weights_list.append(biases)

        with tf.name_scope('fc2'):
            weights = _get_weights(name='fc2', shape=[32, self.env.action_space.n])
            biases = _get_biases(name='fc2', shape=[self.env.action_space.n])
            fc2 = tf.nn.softmax(tf.matmul(fc1, weights) + biases)
            weights_list.append(weights)
            weights_list.append(biases)
            
        logits = fc2
        return logits, weights_list


    def loss(self, action, y, q_values):
        action_one_hot = tf.one_hot(action, self.env.action_space.n, 1.0, 0.0)
        # 行動のQ値の計算
        # tf.reduce_sum() 次元間の要素の合計
        q_value = tf.reduce_sum(tf.multiply(q_values, action_one_hot), reduction_indices=1)
        # エラークリップ
        error = tf.abs(y - q_value)
        # 0.0 ~ 1.0の間に正規化
        quadratic = tf.clip_by_value(error, 0.0, 1.0)
        linear = error - quadratic
        # 誤差関数
        # tf.square() -> ２乗
        loss = tf.reduce_mean(0.5 * tf.square(quadratic) + linear)
        return loss
    

    def training(self, loss, q_network_weights):
        # 最適化
        # global_step = tf.Variable(0, name='global_step', trainable=False)
        optimizer = tf.train.RMSPropOptimizer(LEARNING_RATE, momentum=MOMENTUM, epsilon=MIN_GRAD)
        # 誤差最小化
        gradient_update = optimizer.minimize(loss, var_list=q_network_weights)
        return gradient_update


    def sampling(self):
        return random.sample(list(self.experience_memory), self.batch_size)


    def preprocessing(self, observation, last_observation):
        processed_observation = np.maximum(observation, last_observation)
        resize = cv2.resize(cv2.cvtColor(processed_observation, cv2.COLOR_RGB2GRAY), (84, 84))
        return np.uint8(resize) * 255
        

    def get_initial_state(self, observation, last_observation):
        processed_observation = self.preprocessing(observation, last_observation)
        state = [processed_observation for _ in range(WINDOW_LENGTH)]
        return np.stack(state, axis=0)


    def fit(self):
        # episode
        action_steps = 0

        # モデルの設定
        # observation -> (210, 160, 3)
        preprocess_x = tf.placeholder(tf.float32, shape=self.env.observation_space.shape)
        preprocess_x = tf.reshape(preprocess_x, shape=[100800])

        x = tf.placeholder(tf.float32, shape=(None, WINDOW_LENGTH, 100800))
        s = tf.placeholder(tf.float32, shape=(None, WINDOW_LENGTH, 100800))
        v_action = tf.placeholder(tf.int64, [None])
        y = tf.placeholder(tf.float32, [None])
        # q networkの構築
        q_network, q_network_weights = self.inference(x, 'q_network')
        # target networkの構築
        target_network, target_network_weights = self.inference(s, 'target_network')
        # target networkの更新
        self.update_target_network = [target_network_weights[i].assign(q_network_weights[i]) \
                                      for i in range(len(target_network_weights))]
        loss = self.loss(v_action, y, q_network)
        train = self.training(loss, q_network_weights)

        self.sess = tf.Session()
        global_op = tf.global_variables_initializer()
        self.sess.run([global_op])
        self.sess.run([self.update_target_network])

        with self.sess.as_default():
            for n_episode in range(FLAGS.max_episode):
                observation = self.env.reset()

                for _ in range(random.randint(1, 30)):
                    last_observation = observation.copy()
                    observation, _, _, _ = self.env.step(0)

                state0 = self.get_initial_state(observation, last_observation)

                # step
                done = False
                step = 0
                while not done:
                    last_observation = observation.copy()
                    # observationを画面へ表示
                    self.env.render()
                    # actionを適当に決める
                    # state0はここで利用する予定
                    action = self.env.action_space.sample()
                    # actionを渡してstepし、次のobservation(s`)や報酬を受け取る
                    observation, reward, done, info = self.env.step(action)
                    preprocessed_observation = np.reshape(self.preprocessing(observation, last_observation), (1, 84, 84))
                    self.experience_memory.append(self.experience(state0=state0, action=action,
                                                                  reward=reward, state1=preprocessed_observation,
                                                                  terminal1=done))
                    if LIMIT_EXPERIENCE < len(self.experience_memory):
                        self.experience_memory.popleft()

                    # 次の状態を作成する
                    state0 = np.append(state0[1:, :, :], preprocessed_observation, axis=0)

                    # action stepsがmemoryサイズを超えないと学習させない
                    # memoryサイズがある程度ないとmini batchが作れないため
                    if INITIAL_REPLAY_SIZE < action_steps:
                        if action_steps % TRAIN_INTERVAL is 0:
                            # training
                            mini_batch = self.sampling()
                            state0_batch = []
                            action_batch = []
                            reward_batch = []
                            state1_batch = []
                            terminal1_batch = []
                            for batch in mini_batch:
                                state0_batch.append(batch.state0)
                                action_batch.append(batch.action)
                                reward_batch.append(batch.reward)
                                state1_batch.append(batch.state1)
                                terminal1_batch.append(batch.terminal1)

                            terminal1_batch = np.array(terminal1_batch, dtype=np.int8) + 0
                            calc_pass_q_value = np.array(state1_batch, dtype=np.float32)
                            q_value_batch = q_network.eval(feed_dict={x: calc_pass_q_value})
                            y_batch = reward_batch + (1 - terminal1_batch) * GAMMA * np.max(q_value_batch, axis=1)

                            state_batch = np.array(state0_batch, dtype=np.float32).reshape(32, 1, 210*160*3)
                            _loss, _ = self.sess.run([loss, gradient_update], feed_dict={
                                s: state_batch,
                                v_action: action_batch,
                                y: y_batch})
                            print(_loss)


                        if action_steps % TARGET_UPDATE_INTERVAL is 0:
                            # target_networkのupdate
                            self.sess.run([self.update_target_network])

                    action_steps += 1
                    step += 1

                    # episode終了
                    if done:
                        print('Episode finished after %d timesteps' % (step + 1))

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
