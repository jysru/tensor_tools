import numpy as np
import torch as th
import tensorflow as tf



def pearson_tf(y_true: tf.Tensor, y_pred: tf.Tensor, squared: bool = False, inversed: bool = False) -> tf.Tensor:
    y_true, y_pred = tf.math.abs(y_true), tf.math.abs(y_pred)
    if squared:
        y_true, y_pred = tf.math.square(y_true), tf.math.square(y_pred)
    s = tf.math.reduce_sum((y_true - tf.math.reduce_mean(y_true)) * (y_pred - tf.math.reduce_mean(y_pred)) / tf.cast(tf.size(y_true), tf.float32))
    p = s / (tf.math.reduce_std(y_true) * tf.math.reduce_std(y_pred))
    if inversed:
        p = 1 - p
    return p





