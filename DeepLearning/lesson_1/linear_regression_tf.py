# -*- coding:utf-8 -*-
# __version__ = "TensorFlow_v2.2.0"
# # TensorFlow线性回归代码示例

# In[1]:

from __future__ import print_function
import tensorflow as tf
import numpy
import matplotlib.pyplot as plt

rng = numpy.random

# In[2]:
# Hyper Parameters, 超参数
learning_rate = 0.01
training_epochs = 1000  # 训练迭代次数
display_step = 50

# In[3]:
# Training Data，训练数据
train_X = numpy.asarray([3.3, 4.4, 5.5, 6.71, 6.93, 4.168, 9.779, 6.182, 7.59, 2.167,
                         7.042, 10.791, 5.313, 7.997, 5.654, 9.27, 3.1])
train_Y = numpy.asarray([1.7, 2.76, 2.09, 3.19, 1.694, 1.573, 3.366, 2.596, 2.53, 1.221,
                         2.827, 3.465, 1.65, 2.904, 2.42, 2.94, 1.3])
n_samples = train_X.shape[0]
print(f"n_samples = {n_samples}")

# In[5]:
tf.compat.v1.disable_eager_execution()

# tf Graph Input，tf图输入
X = tf.compat.v1.placeholder("float")
Y = tf.compat.v1.placeholder("float")

# Set model weights，初始化网络模型的权重
W = tf.Variable(rng.randn(), name="weight")  # 权重
b = tf.Variable(rng.randn(), name="bias")  # 偏移

# In[6]:
# Construct a linear model，构造线性模型
pred = tf.add(tf.multiply(X, W), b)

# In[7]:
# Mean squared error，损失函数：均方差
# 问题：这里的均方差多除以了一个2，是什么意思？
# 答案：½是一个常量，这样是为了在求梯度的时候，二次方乘下来的2就和这里的½抵消了，自然就没有多余的常数系数，方便后续的计算，同时对结果不会有影响.
cost = tf.reduce_sum(tf.pow(pred - Y, 2)) / (2 * n_samples)
# Gradient descent， 优化方式：梯度下降
# optimizer = tf.train.GradientDescentOptimizer(learning_rate).minimize(cost) # 历史版本
optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate).minimize(cost)

# In[8]:
# Initialize the variables (i.e. assign their default value)，初始化所有图节点参数
init = tf.compat.v1.global_variables_initializer()

# In[9]:
# Start training，开始训练
with tf.compat.v1.Session() as sess:
    sess.run(init)

    # Fit all training data
    for epoch in range(training_epochs):
        for (x, y) in zip(train_X, train_Y):
            sess.run(optimizer, feed_dict={X: x, Y: y})

        # Display logs per epoch step
        if (epoch + 1) % display_step == 0:
            c = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
            print("Epoch:", '%04d' % (epoch + 1), "cost=", "{:.9f}".format(c), "W=", sess.run(W), "b=", sess.run(b))

    print("Optimization Finished!")
    training_cost = sess.run(cost, feed_dict={X: train_X, Y: train_Y})
    print(f"Training cost={training_cost} W={sess.run(W)} b={sess.run(b)}")

    # Graphic display
    plt.plot(train_X, train_Y, 'ro', label='Original data')
    plt.plot(train_X, sess.run(W) * train_X + sess.run(b), label='Fitted line')
    plt.legend()
    plt.show()
