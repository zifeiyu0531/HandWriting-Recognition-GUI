# coding:utf-8

import tensorflow as tf
import numpy as np
from PIL import Image
from . import backward
from . import forward
from cv import pre


def restore_model(testPicArr):
    with tf.Graph().as_default() as tg:
        x = tf.placeholder(tf.float32, [None, forward.INPUT_NODE])
        y = forward.forward(x, None)
        preValue = tf.argmax(y, 1)

        variable_averages = tf.train.ExponentialMovingAverage(
            backward.MOVING_AVERAGE_DECAY)
        variables_to_restore = variable_averages.variables_to_restore()
        saver = tf.train.Saver(variables_to_restore)

        with tf.Session() as sess:
            ckpt = tf.train.get_checkpoint_state(
                backward.MODEL_SAVE_PATH)
            if ckpt and ckpt.model_checkpoint_path:
                saver.restore(sess, ckpt.model_checkpoint_path)

                preValue = sess.run(preValue, feed_dict={x: testPicArr})
                return preValue
            else:
                print("No checkpoint file found")
                return -1


def pre_pic(picName):
    # 调用图片处理
    im_arr = np.array(pre.image_process(picName))
    threshold = 50
    for i in range(28):
        for j in range(28):
            im_arr[i][j] = 255 - im_arr[i][j]
            if (im_arr[i][j] < threshold):
                im_arr[i][j] = 0
            else:
                im_arr[i][j] = 255

    nm_arr = im_arr.reshape([1, 784])
    nm_arr = nm_arr.astype(np.float32)
    img_ready = np.multiply(nm_arr, 1.0/255.0)

    return img_ready


def application(path):
    testPicArr = pre_pic(path)
    preValue = restore_model(testPicArr)
    print(preValue[0])
    return preValue[0]
