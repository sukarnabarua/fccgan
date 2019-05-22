from __future__ import absolute_import
from __future__ import print_function
from matplotlib import pyplot as plt

import math
import os
import multiprocessing as mp
from subprocess import call
import warnings
import numpy as np
import scipy.io as sio
from tqdm import tqdm
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, roc_auc_score
from sklearn.linear_model import LogisticRegressionCV
from sklearn.preprocessing import scale
import keras.backend as K
from keras.datasets import mnist, cifar10
from keras.utils import np_utils
from keras.models import Sequential
from keras.layers import Dense, Dropout, Activation, Flatten
from keras.layers import Conv2D, MaxPooling2D
from keras.regularizers import l2
import tensorflow as tf
from scipy.spatial.distance import cdist
from keras import regularizers
from keras.models import Model
from sklearn.decomposition import PCA

def get_bn_axis():
    if K.image_dim_ordering() == 'th':
        return 1
    else:
        return -1

# lid of a single query point x
def mle_single(data, x, k=20):
    data = np.asarray(data, dtype=np.float32)
    x = np.asarray(x, dtype=np.float32)
    # print('x.ndim',x.ndim)
    if x.ndim == 1:
        x = x.reshape((-1, x.shape[0]))
    # dim = x.shape[1]

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))
    a = cdist(x, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a[0]


# lid of a batch of query points X
def mle_batch(data, batch, k):
    data = np.asarray(data, dtype=np.float32)
    batch = np.asarray(batch, dtype=np.float32)

    k = min(k, len(data) - 1)
    f = lambda v: - k / np.sum(np.log(v / v[-1]))
    a = cdist(batch, data)
    a = np.apply_along_axis(np.sort, axis=1, arr=a)[:, 1:k + 1]
    a = np.apply_along_axis(f, axis=1, arr=a)
    return a

def get_lids_orig_gan_random_batch(X_orig, X_art, k=10, batch_size=100):
    n_batches = int(np.ceil(X_orig.shape[0] / float(batch_size)))
    i_batch = np.random.randint(0, n_batches)
    X_act = X_orig
    X_gan = X_art
    start = i_batch * batch_size
    end = np.minimum(len(X_act), (i_batch + 1) * batch_size)
    n_feed = end - start
    X_act = X_act[start:end] #extract a batch from original images
    X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))
    # X_gan is an array of batch_size samples, so no need to extract a batch
    # extract a batch from artifical images, remember X_gan is already an array of batch size
    X_gan = X_gan[0:(end-start)]
    X_gan = np.asarray(X_gan, dtype=np.float32).reshape((n_feed, -1))
    # Estimate LID for original images
    lid_batch = mle_batch(X_act, X_act, k=k).reshape(-1,1)
    # Estimate LID for artificial images wrt original dataset
    lid_batch_gan = mle_batch(X_act, X_gan, k=k).reshape(-1,1)
    return lid_batch, lid_batch_gan


def get_lids_orig_gan(X_orig, X_art, k=10, batch_size=100):
    lid_dim = 1 #currently we are calculating LIDs for only one dimension
    print("Number of layers to estimate: ", lid_dim)

    def estimate(i_batch, X_act, X_gan):
        start = i_batch * batch_size
        end = np.minimum(len(X_act), (i_batch + 1) * batch_size)
        n_feed = end - start
        X_act = X_act[start:end] #extract ith batch from original images
        X_act = np.asarray(X_act, dtype=np.float32).reshape((n_feed, -1))
        # print("X_act: ", X_act.shape)
        X_gan = X_gan[start:end] #extract ith batch from artifical images
        X_gan = np.asarray(X_gan, dtype=np.float32).reshape((n_feed, -1))
        # Estimate LID for original images
        lid_batch = mle_batch(X_act, X_act, k=k)
        # print("lid_batch: ", lid_batch.shape)
        # Estimate LID for artificial images wrt original dataset
        lid_batch_gan = mle_batch(X_act, X_gan, k=k)
        # print("lid_batch_noisy: ", lid_batch_noisy.shape)
        return lid_batch, lid_batch_gan

    lids = []
    lids_gan = []
    n_batches = int(np.ceil(X_orig.shape[0] / float(batch_size)))
    for i_batch in tqdm(range(n_batches)):
        lid_batch, lid_batch_gan = estimate(i_batch, X_orig, X_art)
        lids.extend(lid_batch)
        lids_gan.extend(lid_batch_gan)
        # print("lids: ", lids.shape)
        # print("lids_gan: ", lids_gan.shape)
    lids = np.asarray(lids, dtype=np.float32).reshape(-1,1)
    lids_gan = np.asarray(lids_gan, dtype=np.float32).reshape(-1,1)
    return lids, lids_gan

def compute_mle(dists):
    epsilon = 1.0e-10
    dists = dists + epsilon
    k = len(dists)
    log_vals = np.log( dists/(dists[-1] + epsilon) )
    log_sum = np.sum(log_vals)
    lid = -1.0*k/(log_sum)
    return lid
    #f = lambda v: - k / np.sum(np.log(v / v[-1]))

def compute_mle_old(dists):
    dists = dists
    dists = dists[dists != 0] #eliminate any 0 to avoid log error
    k = len(dists)
    log_vals = np.log(dists / (dists[-1])) #may be the denoinator  be added with epsilon to get 3.98 for mnist real
    log_vals[log_vals == -float('inf')] = 0 #still if any inf, remove
    log_sum = np.sum(log_vals)
    lid = -1.0*k/log_sum
    return lid
    #f = lambda v: - k / np.sum(np.log(v / v[-1]))

def compute_lid_batch(X_ref, X, k, batch_size):
    #computes LID for every row vector in X with respect to X_ref set of vectors
    batch_selection = np.random.choice(X_ref.shape[0], batch_size, False)
    X_ref_batch = X_ref[batch_selection]
    dist = cdist(X, X_ref_batch)
    sorted = np.sort(dist, axis=1)
    least_k = sorted[:, 0:k-1]
    lid_batch = np.apply_along_axis(func1d=compute_mle, axis = 1, arr=least_k)
    return lid_batch

def compute_lid_set(X_real, X_art, k, batch_size):
    X_real = X_real.reshape(X_real.shape[0], -1)
    X_art = X_art.reshape(X_art.shape[0], -1)
    n_batches = int(np.ceil(X_real.shape[0]/batch_size))
    lid_real_vals = []
    lid_art_vals = []
    for i in range(n_batches):
        start = i*batch_size
        end = np.min([(i+1)*batch_size, X_real.shape[0]])
        X_real_batch = X_real[start:end]
        X_art_batch = X_art[start:end]
        lid_real_batch = compute_lid_batch(X_real, X_real_batch, k, batch_size)
        lid_art_batch = compute_lid_batch(X_real, X_art_batch, k, batch_size)
        lid_real_vals.extend(lid_real_batch)
        lid_art_vals.extend(lid_art_batch)
    return (np.array(lid_real_vals), np.array(lid_art_vals))

def compute_lid(X, Y, k, batch_size):
    #Computes LIDs for every element of Y with respect to X using given k and batchsize
    #Do not assume that X and Y have same number of rows, but other dimensions must be same
    X = X.reshape(X.shape[0], -1)
    Y = Y.reshape(Y.shape[0], -1)
    n_batches = int(np.ceil(Y.shape[0]/batch_size))
    lid_vals = []
    for i in range(n_batches):
        start = i*batch_size
        end = np.min([(i+1)*batch_size, Y.shape[0]])
        Y_batch = Y[start:end]
        lid_batch = compute_lid_batch(X, Y_batch, k, batch_size)
        lid_vals.extend(lid_batch)
    return np.array(lid_vals)


def scale_value(X, to_min_max, verbose=True):
    from_min_max = get_min_max(X)
    print('from: ', from_min_max)
    #Conver to [0, 1]
    if(verbose): print('Scaling image pixels to min max...')
    X = X - from_min_max[0]
    X = X / (from_min_max[1] - from_min_max[0])
    #Convert to to_min_max
    X = X * (to_min_max[1] - to_min_max[0])
    X = X + to_min_max[0]
    if(verbose): print('Min max: ', get_min_max(X))
    return X


def get_deep_representations(model, X, layer_idx, batch_size=256):
    # Deep representation layer is at index 2
    # output dimension is 7x7x128
    output_dim = model.layers[layer_idx].output.shape.as_list()
    output_dim[0] = X.shape[0] #first dimension is None from shape, so change it

    get_encoding = K.function(
        [model.layers[0].input, K.learning_phase()],
        [model.layers[layer_idx].output]
    )

    n_batches = int(np.ceil(X.shape[0] / float(batch_size)))
    output = np.zeros(shape=output_dim)
    for i in range(n_batches):
        output[i * batch_size:(i + 1) * batch_size] = \
            get_encoding([X[i * batch_size:(i + 1) * batch_size], 0])[0]

    return output

def get_min_max(inp):
    minv = np.min(inp, axis=0)
    minv = np.min(minv)
    minv = np.min(minv)
    minv = np.min(minv)

    maxv = np.max(inp, axis=0)
    maxv = np.max(maxv)
    maxv = np.max(maxv)
    maxv = np.max(maxv)

    return [minv, maxv]

def show_images(image_array):
    size = image_array.shape[0]
    rows = int(math.sqrt(size))
    cols = int(size*1.0/rows)
    plt.figure(figsize=(6, 6))
    for i in range(rows*cols):
        plt.subplot(rows, cols, i + 1)
        img = image_array[i, :] * 0.5 + 0.5
        img = img.reshape((28, 28))
        plt.tight_layout()
        plt.imshow(img, cmap='gray')
        plt.axis('off')
    plt.show()

if __name__ == '__main__':
    start_vals = [200, 300, 400, 500, 600, 700, 800, 900, 1000]
    mean_lids = []
    X_real = np.random.normal(200, 100, size=(1000, 2, 2))
    X_art = np.random.normal(100, 100, size=(1000, 2, 2))
    reals = []
    arts = []
    for start in start_vals:
        (lid_reals, lid_arts) = compute_lid_set(X_real, X_art, k=100, batch_size=start)
        reals.extend( [np.mean(lid_reals)] )
        arts.extend( [np.mean(lid_arts)] )

    print(reals)
    fig = plt.figure()
    plt.plot(start_vals, reals, 'b')
    plt.plot(start_vals, arts, 'g')
    plt.show()
