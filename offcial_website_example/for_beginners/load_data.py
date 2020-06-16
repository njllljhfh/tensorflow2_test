# -*- coding:utf-8 -*-
""" __official_website__ = <https://tensorflow.google.cn/tutorials/load_data/csv> """
from __future__ import absolute_import, division, print_function, unicode_literals
import ssl
import functools

import os

import numpy as np
import tensorflow as tf
import logging
from log_config import settings

logger = logging.getLogger(__name__)
ssl._create_default_https_context = ssl._create_unverified_context  # 关掉ssl验证

"""Load CSV data"""

"""
This tutorial provides an example of how to load CSV data from a file into a tf.data.Dataset.

The data used in this tutorial are taken from the Titanic passenger list. 
The model will predict the likelihood a passenger survived 
based on characteristics like age, gender, ticket class, and whether the person was traveling alone.
"""

TRAIN_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/train.csv"
TEST_DATA_URL = "https://storage.googleapis.com/tf-datasets/titanic/eval.csv"

# if __name__ == '__main__':
logger.info("""=== Setup ====""")
train_file_path = tf.keras.utils.get_file("train.csv", TRAIN_DATA_URL)
test_file_path = tf.keras.utils.get_file("eval.csv", TEST_DATA_URL)
# Make numpy values easier to read.
np.set_printoptions(precision=3, suppress=True)
print(f"train_file_path = {train_file_path}")
print(f"test_file_path = {test_file_path}")

logger.info("""=== Load data ===""")
# To start, let's look at the top of the CSV file to see how it is formatted.
value = os.system(f'head {train_file_path}')
# You can "load this using pandas"<https://tensorflow.google.cn/tutorials/load_data/pandas_dataframe>,
# and pass the NumPy arrays to TensorFlow.
# If you need to scale up to a large set of files,
# or need a loader
# that integrates with "TensorFlow and tf.data <https://tensorflow.google.cn/guide/data>"
# then use the tf.data.experimental.make_csv_dataset function:

# The only column you need to identify explicitly is the one with the value that the model is intended to predict.
LABEL_COLUMN = 'survived'
LABELS = [0, 1]


# Now read the CSV data from the file and create a dataset.
# (For the full documentation, see tf.data.experimental.make_csv_dataset
# <https://tensorflow.google.cn/api_docs/python/tf/data/experimental/make_csv_dataset>)
def get_dataset(file_path, **kwargs):
    dataset = tf.data.experimental.make_csv_dataset(
        file_path,
        batch_size=5,  # Artificially small to make examples easier to show.
        label_name=LABEL_COLUMN,
        na_value="?",
        num_epochs=1,
        ignore_errors=True,
        **kwargs)
    return dataset


def show_batch(dataset):
    for batch, label in dataset.take(1):
        for key, value in batch.items():
            print("{:20s}: {}".format(key, value.numpy()))


raw_train_data = get_dataset(train_file_path)
raw_test_data = get_dataset(test_file_path)

# Each item in the dataset is a batch, represented as a tuple of (many examples, many labels).
# The data from the examples is organized in column-based tensors (rather than row-based tensors),
# each with as many elements as the batch size (5 in this case).
#
# It might help to see this yourself.
logger.info(f"show_batch(temp_dataset)")
show_batch(raw_train_data)

# As you can see, the columns in the CSV are named.
# The dataset constructor will pick these names up automatically.
# If the file you are working with does not contain the column names in the first line,
# pass them in a list of strings to the "column_names" argument in the make_csv_dataset function.
logger.info("- " * 30)
CSV_COLUMNS = ['survived', 'sex', 'age', 'n_siblings_spouses', 'parch', 'fare', 'class', 'deck', 'embark_town',
               'alone']
temp_dataset = get_dataset(train_file_path, column_names=CSV_COLUMNS)  # 全部列
logger.info(f"show_batch(temp_dataset)")
show_batch(temp_dataset)

# Previous example is going to use all the available columns.
# If you need to omit some columns from the dataset,
# create a list of just the columns you plan to use,
# and pass it into the (optional) "select_columns" argument of the constructor.
logger.info("- " * 30)
SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'class', 'deck', 'alone']
temp_dataset = get_dataset(train_file_path, select_columns=SELECT_COLUMNS)  # 部分列
logger.info(f"show_batch(temp_dataset)")
show_batch(temp_dataset)

logger.info("""=== Data preprocessing ===""")
# CSV文件可以包含各种数据类型。通常，您希望在将数据输入模型之前，将这些混合类型转换为固定长度的向量。
# TensorFlow有一个用于描述常见输入转换的内置系统:
# "tf.feature_column<https://tensorflow.google.cn/api_docs/python/tf/feature_column>",
# 详细信息参见本教程<https://tensorflow.google.cn/tutorials/structured_data/feature_columns>。
#
# 您可以使用任何您喜欢的工具(如nltk或sklearn)对数据进行预处理，并将处理后的输出传递给TensorFlow。
# 在模型内部进行预处理的主要好处是，当您导出模型时，它包含预处理。通过这种方式，您可以将原始数据直接传递给模型。

logger.info("""--- Continuous data """)
# 如果你的数据已经是一个格式化好的数字格式，你可以把数据打包成向量(vector)，然后传递给模型:
SELECT_COLUMNS = ['survived', 'age', 'n_siblings_spouses', 'parch', 'fare']
DEFAULTS = [0, 0.0, 0.0, 0.0, 0.0]
# select_columns: 一个可选的整数索引或字符串列名列表，它指定要选择的CSV数据的列的子集。
#                 如果提供了列名，这些必须与column_names中提供的名称或从文件头行推断出来的名称相对应。
#                 当指定此参数时，将只解析并返回与指定的列对应的CSV列的子集。
#                 使用这种方法可以加快解析速度并降低内存使用。如果指定了 select_columns 和 column_defaults，
#                 那么它们必须具有相同的长度，并且假设 column_defaults 按照增加的列索引的顺序排序。
temp_dataset = get_dataset(train_file_path,
                           select_columns=SELECT_COLUMNS,
                           column_defaults=DEFAULTS)
logger.info(f"show_batch(temp_dataset)")
show_batch(temp_dataset)

example_batch, labels_batch = next(iter(temp_dataset))
# print(f"example_batch = {repr(example_batch)}")
# print(f"labels_batch = {repr(labels_batch)}")

logger.info("- " * 30)


# Here's a simple function that will pack together all the columns:
def pack(features, label):
    return tf.stack(list(features.values()), axis=-1), label


# Apply this to each element of the dataset:
packed_dataset = temp_dataset.map(pack)

for features, labels in packed_dataset.take(1):
    print(features.numpy())
    print()
    print(labels.numpy())
# 如果您有混合数据类型，您可能想要分离出这些简单数字字段。
# tf.feature_column api可以处理它们，但是这会带来一些开销，除非真的有必要，否则应该避免这样做。
# 切换回混合数据集:
logger.info("show_batch(raw_train_data)")
show_batch(raw_train_data)

example_batch, labels_batch = next(iter(temp_dataset))


# 所以定义一个更通用的预处理器，选择一个数字特性的列表，并把它们打包成一列:
class PackNumericFeatures(object):
    def __init__(self, names):
        self.names = names

    def __call__(self, features, labels):
        numeric_features = [features.pop(name) for name in self.names]
        numeric_features = [tf.cast(feat, tf.float32) for feat in numeric_features]
        numeric_features = tf.stack(numeric_features, axis=-1)
        features['numeric'] = numeric_features  # 数值列打包后的数据

        return features, labels


NUMERIC_FEATURES = ['age', 'n_siblings_spouses', 'parch', 'fare']

packed_train_data = raw_train_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

packed_test_data = raw_test_data.map(
    PackNumericFeatures(NUMERIC_FEATURES))

logger.info(f"show_batch(packed_train_data)")
show_batch(packed_train_data)
example_batch, labels_batch = next(iter(packed_train_data))

logger.info(f"==== Data Normalization(数据归一化) ===")
# 连续数据应该被归一化。
import pandas as pd

desc = pd.read_csv(train_file_path)[NUMERIC_FEATURES].describe()
print(f"desc = \n{desc}")
logger.info("- " * 30)

MEAN = np.array(desc.T['mean'])
STD = np.array(desc.T['std'])


def normalize_numeric_data(data, mean, std):
    # Center the data
    return (data - mean) / std


# Now create a numeric column. The tf.feature_columns.numeric_column API accepts a normalizer_fn argument,
# which will be run on each batch.
# Bind the MEAN and STD to the normalizer fn using functools.partial.
# See what you just created.
normalizer = functools.partial(normalize_numeric_data, mean=MEAN, std=STD)

numeric_column = tf.feature_column.numeric_column('numeric', normalizer_fn=normalizer, shape=[len(NUMERIC_FEATURES)])
numeric_columns = [numeric_column]
print(f"numeric_column = {numeric_column}")

# When you train the model, include this feature column to select and center this block of numeric data:
logger.info("- " * 30)
logger.info(f"example_batch['numeric']: ")
print(f"{repr(example_batch['numeric'])}")

numeric_layer = tf.keras.layers.DenseFeatures(numeric_columns)
print(f"numeric_layer(example_batch).numpy()={repr(numeric_layer(example_batch).numpy())}")
# The mean based normalization used here requires knowing the means of each column ahead of time.

logger.info("=== Categorical data ===")
# CSV数据中的一些列是分类列。也就是说，内容应该是有限的选项之一。
# Use the tf.feature_column API to create a collection
# with a tf.feature_column.indicator_column for each categorical column.
CATEGORIES = {
    'sex': ['male', 'female'],
    'class': ['First', 'Second', 'Third'],
    'deck': ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J'],
    'embark_town': ['Cherbourg', 'Southhampton', 'Queenstown'],
    'alone': ['y', 'n']
}
categorical_columns = []
for feature, vocab in CATEGORIES.items():
    cat_col = tf.feature_column.categorical_column_with_vocabulary_list(
        key=feature, vocabulary_list=vocab)
    categorical_columns.append(tf.feature_column.indicator_column(cat_col))
print(f"categorical_columns = \n{repr(categorical_columns)}")

categorical_layer = tf.keras.layers.DenseFeatures(categorical_columns)
print(categorical_layer(example_batch).numpy()[0])
# 稍后在构建模型时，这将成为数据处理输入的一部分。

logger.info(f"=== Combined preprocessing layer ===")
# Add the two feature column collections and pass them to a tf.keras.layers.DenseFeatures
# to create an input layer that will extract and preprocess both input types:
# tf.keras.layers.DenseFeatures<https://tensorflow.google.cn/api_docs/python/tf/keras/layers/DenseFeatures>
preprocessing_layer = tf.keras.layers.DenseFeatures(categorical_columns + numeric_columns)
print(preprocessing_layer(example_batch).numpy()[0])

logger.info("=== Build the model ===")
# Build a tf.keras.Sequential <https://tensorflow.google.cn/api_docs/python/tf/keras/Sequential>,
# starting with the preprocessing_layer.
model = tf.keras.Sequential([
    preprocessing_layer,
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(128, activation='relu'),
    tf.keras.layers.Dense(1),
])

model.compile(
    loss=tf.keras.losses.BinaryCrossentropy(from_logits=True),
    optimizer='adam',
    metrics=['accuracy'])

logger.info(f"=== Train, evaluate, and predict ===")
# 现在可以实例化和训练模型了。
train_data = packed_train_data.shuffle(500)
test_data = packed_test_data
model.fit(train_data, epochs=20)

# Once the model is trained, you can check its accuracy on the test_data set.
test_loss, test_accuracy = model.evaluate(test_data)
print('\n\nTest Loss {}, Test Accuracy {}'.format(test_loss, test_accuracy))
logger.info("- " * 30)

# Use tf.keras.Model.predict to infer labels on a batch or a dataset of batches.
# tf.keras.Model.predict <https://tensorflow.google.cn/api_docs/python/tf/keras/Model#predict>
predictions = model.predict(test_data)

# Show some results
for prediction, survived in zip(predictions[:10], list(test_data)[0][1][:10]):
    prediction = tf.sigmoid(prediction).numpy()
    print("Predicted survival: {0:>8.2%}".format(prediction[0]),
          " | Actual outcome: ",
          ("SURVIVED" if bool(survived) else "DIED"))
