# -*- coding:utf-8 -*-
from pprint import pprint

import tf as tf

if __name__ == '__main__':
    import numpy as np

    x_in = np.array([[
        [[2, 6], [1, 6], [2, 6], [0, 6], [1, 6]],
        [[1, 6], [3, 6], [2, 6], [2, 6], [3, 6]],
        [[1, 6], [1, 6], [3, 6], [3, 6], [0, 6]],
        [[2, 6], [2, 6], [0, 6], [1, 6], [1, 6]],
        [[0, 6], [0, 6], [3, 6], [1, 6], [2, 6]],
        [[0, 6], [0, 6], [3, 6], [1, 6], [2, 6]], ]])
    print(x_in)
    print(x_in.shape)
    print("- " * 30)

    kernel_in = np.array([
        [[[2, 0.1, 6]], [[3, 0.2, 6]]],
        [[[0, 0.3, 6]], [[1, 0.4, 6]]], ])
    print(kernel_in)
    print(kernel_in.shape)

    import tensorflow as tf

    x = tf.Variable(tf.random_normal([7 * 7 * 64, 1024]))
    print(x.get_shape())

    print("- " * 30)
    # t1 = tf.Variable([1, 2, 3, 4, 5])
    t1 = tf.Variable([True, False, True, False, False])
    t2 = tf.cast(t1, dtype=tf.float32)

    print('t1: {}'.format(t1))
    print('t2: {}'.format(t2))

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())
        sess.run(t2)
        print(t2.eval())
