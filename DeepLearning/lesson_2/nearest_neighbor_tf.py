# -*- coding:utf-8 -*-
# __origin_code__ = <https://github.com/ionvision/DeepLearningCourseCodes/blob/master/02_Logistic_regression_and_multilayer_perceptron/nearest_neighbor_tf.py>
# 原码是tf1.0版本，本文件的运行环境是tf2.2版本，所以修改了部分代码。
# # K近邻算法, TensorFlow示例
# ### K Nearest Neighbor (KNN) Example

# In[1]:

from __future__ import print_function
import numpy as np
import tensorflow as tf

tf.compat.v1.disable_eager_execution()  # 保证sess.run()能够正常运行

# Import MNIST data, 准备MNIST数据输入
# from tensorflow.examples.tutorials.mnist import input_data
#
# mnist = input_data.read_data_sets("/tmp/data/", one_hot=True)
mnist = tf.keras.datasets.mnist

# Load and prepare the MNIST dataset.
# Convert the samples from integers to floating-point numbers:
(x_train, y_train), (x_test, y_test) = mnist.load_data()
print(f"lent(x_train) = {len(x_train)}")

# In[2]:

# In this example, we limit mnist data，限制候选比较数据的数目
# Xtr, Ytr = mnist.train.next_batch(5000)  # 5000 for training (nn candidates)
# Xte, Yte = mnist.test.next_batch(200)  # 200 for testing
Xtr, Ytr = x_train[:5000, :], y_train[:5000]  # 5000 for training (nn candidates)
Xtr = Xtr.reshape([5000, 28 * 28])
Xte, Yte = x_test[0:200, :], y_test[0:200]  # 200 for testing
Xte = Xte.reshape([200, 28 * 28])
print(f"lent(Xtr) = {len(Xtr)}")
print(f"lent(Xtr.shape) = {Xtr.shape}")
print(f"lent(Ytr) = {len(Ytr)}")
print(f"lent(Ytr.shape) = {Ytr.shape}")
print(f"lent(Xte) = {len(Xte)}")
print(f"lent(Xte.shape) = {Xte.shape}")
print(f"lent(Yte) = {len(Yte)}")
print(f"lent(Yte.shape) = {Yte.shape}")
print(f"Yte = {Yte}")

# tf Graph Input，TensorFlow图模型的输入定义
xtr = tf.compat.v1.placeholder("float", [None, 784])
xte = tf.compat.v1.placeholder("float", [784])

# Nearest Neighbor calculation using L1 Distance，最小近邻算法的L1曼哈顿距离计算
# Calculate L1 Distance
# distance = tf.compat.v1.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), reduction_indices=1)  # 损失函数(L1距离)-历史写法
# axis=1 按行求和
distance = tf.reduce_sum(tf.abs(tf.add(xtr, tf.negative(xte))), axis=1)  # 损失函数(L1距离)

# Prediction: Get min distance index (Nearest neighbor)，最小距离近邻的预测
# distance 是一个一维的列向量吗？？？
print(f"distance.shape = {distance.shape}")
pred = tf.compat.v1.arg_min(distance, 0)  # 优化方式(最小L1距离)
print(f"pred.shape = {pred.shape}")  # 是一个整数？？？

accuracy = 0.

# Initialize the variables (i.e. assign their default value)，初始化所有参数
init = tf.compat.v1.global_variables_initializer()

# In[3]:

# Start training
with tf.compat.v1.Session() as sess:
    sess.run(init)

    # loop over test data
    for i in range(len(Xte)):
        # Get nearest neighbor
        # nn_index:是训练集中，与当前输入的这一条验证集数据，距离最近的训练数据的在训练集中的index。
        # 该过程，要将验证集的者一条数据，与训练集中的全部数据一一对比。
        nn_index = sess.run(pred, feed_dict={xtr: Xtr, xte: Xte[i, :]})  # Xtr:训练集, Xte[i, :]:验证集中的一条数据
        print(f"nn_index = {nn_index}")
        # Get nearest neighbor class label and compare it to its true label
        # print(f"Ytr[nn_index] = {Ytr[nn_index]}")
        # Ytr[nn_index] 是一个数字<class 'numpy.uint8'>:表示这条训练数据的类别（0-9）
        # Yte[i] 是一个数字<class 'numpy.uint8'>:表示这条验证数据的类别（0-9）
        print("Test", i, "Prediction:", Ytr[nn_index], "True Class:", Yte[i])

        # Calculate accuracy
        # if np.argmax(Ytr[nn_index]) == np.argmax(Yte[i]):
        if Ytr[nn_index] == Yte[i]:
            accuracy += 1. / len(Xte)
    print("Done!")
    print("Accuracy:", accuracy)
