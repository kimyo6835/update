from __future__ import division, print_function, unicode_literals

import matplotlib.pyplot as plt
import tensorflow.contrib.slim as slim
from tada import wires_functions
from tada import capsule_functions
import tensorflow as tf
import numpy as np
import imageio as io
import scipy.misc

checkpoint_path="./180424_test_epoch5"
tf.reset_default_graph()

epsilon = 1e-5
batch_size = 4
imsize = 32

x = tf.placeholder(shape=[None, imsize, imsize, 1], dtype=tf.float32, name="X")

# ------- ReLU layer -------- #
# kernel size = [5, 5]
# out_channels = 16
# stride = 1 with padding ('same')
ReLU_conv1 = capsule_functions.conv2d(x, 5, 16, 1, 'SAME', 'ReLU_conv1', True, tf.nn.relu)

# ------- PrimaryCaps layer -------- #
# kernel size = [5, 5]
# out_capsules = 16
# stride = 1 with padding ('same')
# pose shape = [4, 4]
PrimaryCaps = capsule_functions.primary_caps(ReLU_conv1, 5, 16, 1, 'SAME', (4, 4), "PrimaryCaps")

# spatial dimension = image dimension
# ------- ConvCaps layer -------- #
# i = number of input capsules
# o = number of output capsules
# convolution operation kernel, [kh, kw, i, o] = (3, 3, 16, 2)
# stride = 1 (1,1,1,1) with padding ('same')
# iteration (EM routing) = 3
ConvCaps0, a, b = capsule_functions.conv_capsule(PrimaryCaps, (3, 3, 16, 2), (1, 1, 1, 1), 3, batch_size, "ConvCaps0", 'SAME')
#ConvCaps1 = capsule_functions.class_capsules(ConvCaps0, 1, iterations=3, batch_size=batch_size, name='class_capsules')

# ConvCaps0[0]: [4, 32, 32, 2, 4, 4]

# Select one output capsule
ConvCaps01 = ConvCaps0[0][:,:,:,0,:,:],ConvCaps0[1][:,:,:,0]
ConvCaps1 = ConvCaps01[0][:,:,:,tf.newaxis,:,:],ConvCaps01[1][:,:,:,tf.newaxis]

y = tf.placeholder(shape=[batch_size, imsize, imsize, 17], dtype=tf.float32, name="y")

# ConvCaps1[0]: [4, 32, 32, 1, 4, 4]
# y_final_0: [4, 32, 32, 1, 16] # pose label
y_final_0 = tf.reshape(ConvCaps1[0], [batch_size, imsize, imsize, -1, 16], name="y_final_0")

# ConvCapse1[1]: [4, 32, 32, 1]
# y_final_1: [4, 32, 32, 1, 1] # activation label
y_final_1 = tf.expand_dims(ConvCaps1[1], -1, name="y_final_1")

# Merge pose and activation labels
# y_pred_f: [4, 32, 32, 1, 17]
# y_pred: [4, 32, 32, 17]
y_pred_f = tf.concat([y_final_0, y_final_1], axis=4, name="y_pred_f")
y_pred = tf.squeeze(y_pred_f, axis=3, name="y_pred")

gstep = tf.placeholder(shape=(), dtype=tf.float32, name="gstep")

diff_y = tf.subtract(y, y_pred, name="diff_y")
squa_y = tf.square(diff_y, name="squa_y")

loss = tf.reduce_mean(squa_y, name="loss")

y_group = tf.squeeze(ConvCaps1[1] > 0.5, 3, name="y_group")
y_real = y[:, :, :, 16] > 0.5

correct = tf.equal(y_real, y_group, name="correct")
accuracy = tf.reduce_mean(tf.cast(correct, tf.float32), name="accuracy")

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
#global_step = tf.train.get_or_create_global_step()
#training_op = slim.learning.create_train_op(tf.clip_by_value(loss, 1e-5, 1.0), optimizer, global_step=global_step, clip_gradient_norm=4.0)

saver = tf.train.Saver()
var_init = tf.global_variables_initializer()




init = tf.constant_initializer(0.0)

#checkpoint_path = "./180430_test_epoch5_bs5"
sess=tf.Session()
saver.restore(sess, checkpoint_path)

# ------------ load image and padd it to zero to be consistent with 32x32 tiles ---------------------------------- #
path_to_raw = '/home/gimmedatcudaspeedmmmh/Desktop/NeuronGuys/JIN/capsule_test/idontwanna/sample_stack/d_300_3mg_silver/'

save_path = '/home/gimmedatcudaspeedmmmh/Desktop/NeuronGuys/JIN/capsule_test/idontwanna/sample_stack/d_300_3mg_silver/savefigures/capsule_epoch5_bs4'
#wireindex = np.arange(50, 185 + 1)

wireindex = np.arange(50, 52)

# Convolted image that is not affected by zero padding ( = kerneal sizee - 1)
kernel_size = 5
conv_imsize = imsize - (kernel_size - 1)

for index in wireindex:

    xproj_fn = '%s/x_proj_%d.png' % (save_path, index)
    yproj_fn = '%s/y_proj_%d.png' % (save_path, index)
    intensity_fn = '%s/intensity_%d.png' % (save_path, index)
    xy_add_fn = '%s/xy_proj_add_%d.png' % (save_path, index)

    resized_image, repeat_size = wires_functions.striding_image(path_to_raw, index, imsize, kernel_size, conv_imsize)

    # Create placeholder for the analyzed images
    filtered_x_proj = np.zeros(resized_image.shape)
    filtered_y_proj = np.zeros(resized_image.shape)
    filtered_intensity = np.zeros(resized_image.shape)

    # Start point ( increase by conv_imsize = imsize - zero padding/2 )
    stride_size = conv_imsize
    start_point_x = np.multiply(stride_size, np.arange(0, repeat_size[0])).reshape(-1, batch_size)
    start_point_x1 =start_point_x[0:27+1].reshape(-1, batch_size)
    start_point_x2 = start_point_x[-4:]


    start_point_y = np.multiply(stride_size, np.arange(0, repeat_size[1]))

    k = np.arange(0, start_point_x1.shape[0])
    tile_x = start_point_x1

    for tile_y in start_point_y:
        for kk in k:
            input_tensor = np.zeros((batch_size, imsize, imsize, 1))
            for batch_id in np.arange(0, batch_size):
                input_tensor[batch_id, :, :, 0] = resized_image[tile_x[kk, batch_id]:tile_x[kk, batch_id] + 32,
                                                  tile_y:tile_y + 32]

            # predict this batch
            predictions = sess.run([y_pred], feed_dict={x: input_tensor})
            predictions = np.asarray(predictions).reshape(batch_size, imsize, imsize, 17)

            for batch_id in np.arange(0, batch_size):
                x_idx = tile_x[kk, batch_id]
                y_idx = tile_y

                filtered_x_proj[x_idx + 2:x_idx + 29 + 1, y_idx + 2:y_idx + 29 + 1] = predictions[batch_id, 2:29 + 1,
                                                                                      2:29 + 1, 0]
                filtered_y_proj[x_idx + 2:x_idx + 29 + 1, y_idx + 2:y_idx + 29 + 1] = predictions[batch_id, 2:29 + 1,
                                                                                      2:29 + 1, 1]
                filtered_intensity[x_idx + 2:x_idx + 29 + 1, y_idx + 2:y_idx + 29 + 1] = predictions[batch_id, 2:29 + 1,
                                                                                         2:29 + 1, -1]

tile_x = start_point_x2

for tile_y in start_point_y:
        input_tensor = np.zeros((batch_size, imsize, imsize, 1))
        for batch_id in np.arange(0, batch_size):
                input_tensor[batch_id, :, :, 0] = resized_image[tile_x[batch_id]:tile_x[batch_id]+32, tile_y:tile_y+32]
    
        
        #predict this batch
        predictions=sess.run([y_pred], feed_dict={x:input_tensor})
        predictions=np.asarray(predictions).reshape(batch_size, imsize, imsize, 17)
        
        for batch_id in np.arange(2, batch_size):
            x_idx = tile_x[batch_id]
            y_idx = tile_y            
            
            filtered_x_proj[x_idx+2:x_idx+29+1, y_idx+2:y_idx+29+1] = predictions[batch_id, 2:29+1, 2:29+1, 0]
            filtered_y_proj[x_idx+2:x_idx+29+1, y_idx+2:y_idx+29+1] = predictions[batch_id, 2:29+1, 2:29+1, 1]
            filtered_intensity[x_idx+2:x_idx+29+1, y_idx+2:y_idx+29+1] = predictions[batch_id, 2:29+1, 2:29+1, -1]
     
            

    xy_add = np.abs(filtered_x_proj) + np.abs(filtered_y_proj)

    scipy.misc.imsave(xproj_fn, filtered_x_proj)
    scipy.misc.imsave(yproj_fn, filtered_y_proj)
    scipy.misc.imsave(intensity_fn, filtered_intensity)
    scipy.misc.imsave(xy_add_fn, xy_add)
