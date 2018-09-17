from keras.models import Model
from keras.regularizers import l2
from keras.optimizers import SGD, Adam
from keras.layers import *
from sklearn.metrics import f1_score, jaccard_similarity_score
import tensorflow as tf
import keras.backend as K
import numpy as np

def f1(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return f1_score(y_true_f, y_pred_f)

def jaccard(y_true, y_pred):
    y_true_f = K.flatten(y_true)
    y_pred_f = K.flatten(y_pred)
    return jaccard_similarity_score(y_true_f, y_pred_f)
