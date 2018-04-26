from __future__ import division, print_function, unicode_literals

import tensorflow as tf
import numpy as np
import math
import random
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from tada import wires_functions

# ----------- Generate images of wires ----------- #

x_train, y_train = wires_functions.gen_wires(n_data=1000, length=20, n_wires=1, imsize=32)
x_test, y_test = wires_functions.gen_wires(n_data=1000, length=20, n_wires=1, imsize=32)

# -------- Added to generate image w/o noise -------- #
y_train = np.where(y_train > 0.9, 1.0, 0.0)
y_test = np.where(y_test > 0.9, 1.0, 0.0)


# ----------- Check one of images (train) ----------- #
#check_idx = 15  # random number

#plt.figure()
#plt.imshow(y_train[check_idx, :, :, 15])
#plt.colorbar()
#plt.show()

# ----------- Set parameters for networks ------------- #
imsize = 32
batch_size = 4
###########################################################################################
###########################           Encoder         #####################################
###########################################################################################
# Input Layer: [batch_size, width, height, channels]
# Image size: 32 x 32, One color channel
#input_layer = tf.reshape(features["x"], [-1, 32, 32, 1])
x = tf.placeholder(shape=[None, imsize, imsize, 1], dtype=tf.float32, name="X")
y = tf.placeholder(shape=[None, imsize, imsize], dtype=tf.float32, name="y")

# Convolutional Layer #1
# Computes 32 features using a 5x5 filter with ReLU activation.
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 32, 32, 1]
# Output Tensor Shape: [batch_size, 32, 32, 32]
conv1 = tf.layers.conv2d(
    inputs=x,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=None)

# Batch Normalization
#################################################################################### Q.scope='bn'
h1 = tf.contrib.layers.batch_norm(conv1, center=True, scale=True, is_training=True)
ho1 = tf.nn.relu(h1, 'relu')

# Pooling Layer #1
# First max pooling layer with a 2x2 filter and stride of 2
# Input Tensor Shape: [batch_size, 32, 32, 32]
# Output Tensor Shape: [batch_size, 16, 16, 32]
pool1 = tf.layers.max_pooling2d(inputs=ho1, pool_size=[2, 2], strides=2)

# -----------------------------------------------------------------------------------#
# Convolutional Layer #2
# Computes 64 features using a 5x5 filter with ReLU activation.
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 16, 16, 32]
# Output Tensor Shape: [batch_size, 16, 16, 64]
conv2 = tf.layers.conv2d(
    inputs=pool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=None)

# Batch Normalization
h2 = tf.contrib.layers.batch_norm(conv2, center=True, scale=True, is_training=True)
ho2 = tf.nn.relu(h2, 'relu')

# Pooling Layer #2
# First max pooling layer with a 2x2 filter and stride of 2
# Input Tensor Shape: [batch_size, 16, 16, 64mm]
# Output Tensor Shape: [batch_size, 8, 8, 64]
pool2 = tf.layers.max_pooling2d(inputs=ho2, pool_size=[2, 2], strides=2)

# -----------------------------------------------------------------------------------#
# Convolutional Layer #3
# Computes 128 features using a 5x5 filter with ReLU activation.
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 8, 8, 64]
# Output Tensor Shape: [batch_size, 8, 8, 128]
conv3 = tf.layers.conv2d(
    inputs=pool2,
    filters=128,
    kernel_size=[5, 5],
    padding="same",
    activation=None)

# Batch Normalization
h3 = tf.contrib.layers.batch_norm(conv3, center=True, scale=True, is_training=True)
ho3 = tf.nn.relu(h3, 'relu')

# Pooling Layer #3
# First max pooling layer with a 2x2 filter and stride of 2
# Input Tensor Shape: [batch_size, 8, 8, 128]
# Output Tensor Shape: [batch_size, 4, 4, 128]
pool3 = tf.layers.max_pooling2d(inputs=ho3, pool_size=[2, 2], strides=2)

###########################################################################################
###########################           Decoder         #####################################
###########################################################################################

# Upsampling #1
# Input Tensor Shape: [batch_size, 4, 4, 128]
# Output Tensor Shape: [batch_size, 8, 8, 128]
unpool1 = tf.image.resize_images(pool1, [32, 32])

# Convolutional Layer #1
# Computes 64 features using a 5x5 filter with ReLU activation.
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 8, 8, 128]
# Output Tensor Shape: [batch_size, 8, 8, 64]
dconv1 = tf.layers.conv2d(
    inputs=unpool1,
    filters=64,
    kernel_size=[5, 5],
    padding="same",
    activation=None)

# Batch Normalization
dh1 = tf.contrib.layers.batch_norm(dconv1, center=True, scale=True, is_training=True)
dho1 = tf.nn.relu(h1, 'relu')

# Upsampling #2
# Input Tensor Shape: [batch_size, 8, 8, 64]
# Output Tensor Shape: [batch_size, 16, 16, 64]
unpool2 = tf.image.resize_images(dho1, [16, 16])

# Convolutional Layer #2
# Computes 32 features using a 5x5 filter with ReLU activation.
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 16, 16, 32]
# Output Tensor Shape: [batch_size, 16, 16, 32]
dconv2 = tf.layers.conv2d(
    inputs=unpool2,
    filters=32,
    kernel_size=[5, 5],
    padding="same",
    activation=None)

# Batch Normalization
dh2 = tf.contrib.layers.batch_norm(dconv2, center=True, scale=True, is_training=True)
dho2 = tf.nn.relu(h2, 'relu')

# Upsampling #3
# Input Tensor Shape: [batch_size, 16, 16, 32]
# Output Tensor Shape: [batch_size, 32, 32, 32]
unpool3 = tf.image.resize_images(dho2, [32, 32])

# Convolutional Layer #2
# Computes 32 features using a 5x5 filter with ReLU activation.
# Padding is added to preserve width and height.
# Input Tensor Shape: [batch_size, 32, 32, 32]
# Output Tensor Shape: [batch_size, 32, 32, 2]
dconv3 = tf.layers.conv2d(
    inputs=unpool3,
    filters=2,
    kernel_size=[5, 5],
    padding="same",
    activation=None)

# Batch Normalization
dh3 = tf.contrib.layers.batch_norm(dconv3, center=True, scale=True, is_training=True)
dho3 = tf.nn.relu(dh3, 'relu')

###########################################################################################
###########################       Classification      #####################################
###########################################################################################

pixel_output = tf.nn.softmax(dho3, name="softmax_tensor")
y_pred = pixel_output

class_pred = tf.argmax(input=pixel_output, axis=-1)
#y_pred = y_pred0
#print(y_pred.shape)
#print(y.shape)

#print(y_pred)
#print(y)

diff_y = tf.subtract(y, y_pred, name="diff_y")
squa_y = tf.square(diff_y, name="squa_y")
loss = tf.reduce_mean(squa_y, name="loss")

# Loss
#y_pred = tf.squeeze(y_pred, axis=-1)


#subt = tf.subtract(yyy, y)
#subt = tf.equal(y_pred_boo, y_boo)
#square = tf.square(subt)
#loss = tf.reduce_mean(square, name="loss")
#loss = tf.reduce_mean(tf.cast(subt, tf.float32), name="loss")

y_group = class_pred > 0.5
y_real = y[:, :, :] > 0.5 # I don't think I need this part tho

correct = tf.equal(y_real, y_group, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

# -------------------------- Train with sess ------------------------------- #

total_parameters = 0
for variable in tf.trainable_variables():
    # shape is an array of tf.Dimension
    shape = variable.get_shape()
    variable_parameters = 1
    for dim in shape:
        variable_parameters *= dim.value
    total_parameters += variable_parameters

print(total_parameters)

tf.trainable_variables()

# Tried option: learning_rate = 0.001 / 0.0001
optimizer = tf.train.AdamOptimizer()
training_op = optimizer.minimize(loss, name="training_op")

saver = tf.train.Saver()
var_init = tf.global_variables_initializer()

checkpoint_path = "./180426_CNN"

init = tf.constant_initializer(0.0)

n_epochs = 10
restore_checkpoint = True

n_iterations_per_epoch = y_train.shape[0] // batch_size
n_iterations_validation = y_test.shape[0] // batch_size
best_acc_val = 0.

with tf.Session() as sess:
    if restore_checkpoint and tf.train.checkpoint_exists(checkpoint_path):
        saver.restore(sess, checkpoint_path)
    else:
        var_init.run()
        None

    for epoch in range(n_epochs):

        gsteps = np.array((1. * epoch / (n_epochs - 1)) * 0.1 + 1)
        print(gsteps)

        for iteration in range(1, n_iterations_per_epoch + 1):
            # for iteration in range(1,3):
            X_batch = x_train[(iteration - 1) * batch_size:(iteration + 0) * batch_size, :, :, :]
            y_batch = y_train[(iteration - 1) * batch_size:(iteration + 0) * batch_size, :]
            # Run the training operation and measure the loss:
            _, loss_train = sess.run(
                [training_op, loss],
                feed_dict={x: X_batch,
                           y: y_batch[:, :, :, 15]
                           })

            print("\rIteration: {}/{} ({:.1f}%)  Loss: {:.4f}".format(
                iteration, n_iterations_per_epoch,
                iteration * 100 / n_iterations_per_epoch,
                loss_train),
                end="")

        print()

        # At the end of each epoch,
        # measure the validation loss and accuracy:
        loss_vals = []
        acc_vals = []
        for iteration in range(1, n_iterations_validation + 1):
            X_batch = x_test[(iteration - 1) * batch_size:(iteration + 0) * batch_size, :, :, :]
            y_batch = y_test[(iteration - 1) * batch_size:(iteration + 0) * batch_size, :]
            loss_val, acc_val, big_bois_pred, big_bois_real = sess.run(
                [loss, accuracy, y_pred, y],
                feed_dict={x: X_batch,
                           y: y_batch[:, :, :, 15],
                           })
            loss_vals.append(loss_val)
            acc_vals.append(acc_val)
            print("\rEvaluating the model: {}/{} ({:.1f}%)".format(
                iteration, n_iterations_validation,
                iteration * 100 / n_iterations_validation),
                end=" " * 10)
        loss_val = np.mean(loss_vals)
        acc_val = np.mean(acc_vals)
        print("\rEpoch: {}  Val accuracy: {:.4f}%  Loss: {:.6f}{}".format(
            epoch + 1, acc_val * 100, loss_val,
            " (improved)" if acc_val > best_acc_val else ""))

        # And save the model if it improved:
        if acc_val > best_acc_val:
            save_path = saver.save(sess, checkpoint_path)
            best_acc_val = acc_val