import tensorflow as tf
import numpy as np

# https://github.com/GoogleCloudPlatform/training-data-analyst/blob/master/courses/machine_learning/deepdive/03_tensorflow/labs/a_tfstart.ipynb

#Eager is abilited of default in tensorflow 2.3
#if you want use lazy mode, must disabilited eager mode
tf.compat.v1.enable_eager_execution()


def compute_area(sides):
    a = sides[:,0]
    b = sides[:,1]
    c = sides[:,2]
    s = (a+b+c) / 2

    area = tf.sqrt(s * (s-a) * (s-b) * (s-c))

    return area


area = compute_area(tf.constant([
                                [5.0, 3.0, 7.1],
                                [2.3, 4.1, 4.8]
                                ]))

print(area)