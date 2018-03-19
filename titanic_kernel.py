import pandas as pd
import numpy as np
import tensorflow as tf


def trim_data(data):
    result = data.copy()
    sex_rep = {
        'male': 0,
        'female': 1,
    }
    result['Sex'].replace(sex_rep, inplace=True)
    age_rep = {
        np.nan: result['Age'].mean()
    }
    result['Age'].replace(age_rep, inplace=True)
    del result['PassengerId']
    del result['Name']
    del result['Cabin']
    del result['Ticket']
    del result['Embarked']

    del result['Age']
    del result['Fare']
    return result


tf.reset_default_graph()
sess = tf.Session()

train_raw = trim_data(pd.read_csv('./train.csv'))

msk = np.random.rand(len(train_raw)) < 0.8

# Train data 80%
train_train_raw_no_norm = train_raw[msk]
# train_train_raw = (train_train_raw_no_norm - train_train_raw_no_norm.mean()) / (train_train_raw_no_norm.max()
#                                                                                 - train_train_raw_no_norm.min())
train_train_raw = train_train_raw_no_norm

train_train_labels_raw = train_train_raw_no_norm.copy()[['Survived']]
# train_train_labels_raw['Survived'] = train_train_labels_raw['Survived'].astype(float)
del train_train_raw['Survived']

# Test Train data 20%
train_test_raw = train_raw[~msk]
train_test_labels_raw = train_test_raw.copy()[['Survived']]
del train_test_raw['Survived']

NUM_CLASSES = 1
TRAIN_STEPS = 5000
NUM_FEATURES = len(train_raw.columns) - 1
LEARNING_RATE = 0.5

# print train
# print train_labels
# print train.dtypes
# print train.shape
# print train.describe()

# print train.Age.max()
# print train.groupby(['Embarked']).sum()
# print train['Ticket'].head(1)
# print train.Sex.max()
# print train.dtypes

# Define inputs
x = tf.placeholder(dtype=tf.float32, shape=[None, NUM_FEATURES])
output = tf.placeholder(dtype=tf.float32, shape=[None, NUM_CLASSES])

# Define model
W = tf.Variable(tf.truncated_normal([NUM_FEATURES, NUM_CLASSES]))
b = tf.Variable(tf.zeros([NUM_CLASSES]))
# y = 1 / (1 + tf.exp(-1 * (tf.matmul(x, W) + b)))
y = tf.sigmoid(-1 * (tf.matmul(x, W) + b))

# Define loss and optimizer
loss = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(logits=y, labels=output))
# loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(logits=y, labels=output))
# loss = tf.reduce_mean(tf.square(y - output))
train_step = tf.train.GradientDescentOptimizer(LEARNING_RATE).minimize(loss)

sess.run(tf.global_variables_initializer())

# Train the model
for i in range(TRAIN_STEPS):
    sess.run(train_step, feed_dict={x: train_train_raw, output: train_train_labels_raw})
    if i % 1000 == 0:
        accuracy = 1 - tf.reduce_mean(tf.abs(tf.round(y) - output))
        print("Accuracy %f" % sess.run(accuracy, feed_dict={x: train_train_raw,
                                                        output: train_train_labels_raw}))

# Evaluate the trained model
accuracy = 1 - tf.reduce_mean(tf.abs(tf.round(y) - output))
print "Accuracy on training: "
print("Accuracy %f" % sess.run(accuracy, feed_dict={x: train_train_raw,
                                                    output: train_train_labels_raw}))

print "Accuracy on training testing: "
print("Accuracy %f" % sess.run(accuracy, feed_dict={x: train_test_raw,
                                                    output: train_test_labels_raw}))

test_raw = pd.read_csv('./test.csv')
test = trim_data(test_raw)
result = sess.run(tf.round(y), feed_dict={x: test}).transpose()
report = test_raw.copy()[['PassengerId']]
report['Survived'] = pd.Series(result.flatten()).astype(int)
report.to_csv('./report.csv', sep=',', index=False)
print report
