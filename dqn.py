# -*- coding: utf-8 -*-

import tensorflow as tf


FLAGS = tf.app.flags.FLAGS()
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


def train():
    """Model training."""
    pass


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
