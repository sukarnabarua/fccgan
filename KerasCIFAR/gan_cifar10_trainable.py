from keras.callbacks import Callback
from keras_adversarial import gan_targets, AdversarialModel, simple_gan, normal_latent_sampling, \
    AdversarialOptimizerSimultaneous
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial.legacy import fit
from image_utils import dim_ordering_unfix, dim_ordering_shape
from keras.models import Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dense, AveragePooling2D, Concatenate
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from keras.datasets import cifar10
import numpy as np
import pandas as pd

from inception import get_inception_score
from myutils import get_bn_axis, scale_value


conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

def activation(act, inputs):
    if act == 'relu': return Activation('relu')(inputs)
    elif act == 'leakyrelu': return LeakyReLU(alpha=0.2)(inputs)
    else: return inputs

def cifar10_process(x):
    x = x.astype(np.float32)
    x = scale_value(x, [-1, 1]) #rescale to tanh compatible output
    return x

def cifar10_data(data_size=50000):
    (X_train, Y_train), (xtest, ytest) = cifar10.load_data()

    if(data_size < X_train.shape[0]):
        sel_pct =  data_size / X_train.shape[0]
        print('Selecting percent: ', sel_pct)
        (X_train1, X_train2, Y_train1, Y_train2) = train_test_split(X_train, Y_train, train_size=sel_pct, stratify=Y_train)
    else:
        X_train1 = X_train

    return cifar10_process(X_train1), cifar10_process(xtest)

best_icp = 0

def compute_metric(generator_model):
    global best_icp
    sample_size = 20000
    noise = np.random.normal(size=(sample_size, 100))
    art_images = generator_model.predict(noise)    
    art_images = scale_value(art_images, [-1.0, 1.0])
    art_images = np.transpose(art_images, (0, 3, 1, 2))
    (icp_mean, icp_std) = get_inception_score(art_images)
    if icp_mean > best_icp: best_icp = icp_mean
    print('Inception score: ', icp_mean)


class SaveModelWeights(Callback):
    def __init__(self, g, path):
        self.generator = g
        self.save_path = path

    def on_epoch_end(self, epoch, logs={}):
        save_file_name = self.save_path + '/gen_weight_epoch_' + str(epoch) + '.h5'
        self.generator.save_weights(save_file_name)
        # if (epoch + 1) % 5 == 0 or epoch >= 200: compute_metric(self.generator)


class SaveModelWeightsDiscriminator(Callback):
    def __init__(self, d, path):
        self.discriminator = d
        self.save_path = path

    def on_epoch_end(self, epoch, logs={}):
        save_file_name = self.save_path + '/dis_weight_epoch_' + str(epoch) + '.h5'
        self.discriminator.save_weights(save_file_name)


class ReloadModelWeightsDiscriminator(Callback):
    def __init__(self, d, path):
        self.discriminator = d
        self.save_path = path

    def on_epoch_end(self, epoch, logs={}):
        save_file_name = self.save_path + '/dis_weight_epoch_' + str(0) + '.h5'
        if(epoch > 0):
            print('Reloading epoch 0 dis weights at epoch: ', epoch)
            self.discriminator.load_weights(save_file_name)


usegbn = True
usedbn = True

def setbn(gbn = True, dbn = True):
    global usegbn
    global usedbn
    usegbn = gbn
    usedbn = dbn


def dcgan_generator(bnmode=0):
    global usegbn, conv_init

    d_input = Input(shape=(100,))

    L = Reshape(target_shape=dim_ordering_shape((100, 1, 1)))(d_input)

    L = Conv2DTranspose(filters=256, kernel_size=4, strides=1, use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=128, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=64, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=3, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    d_output = Activation('tanh')(L)

    return Model(d_input, d_output)
    global usegbn, conv_init

    d_input = Input(shape=(100,))

    L = Reshape(target_shape=dim_ordering_shape((100, 1, 1)))(d_input)

    L = Conv2DTranspose(filters=256, kernel_size=4, strides=1, use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    #Extra start
    L = Conv2D(filters=128, kernel_size=4, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2D(filters=256, kernel_size=4, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=128, kernel_size=3, strides=1, use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)
    #Extra End

    L = Conv2DTranspose(filters=128, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=64, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=3, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    d_output = Activation('tanh')(L)

    return Model(d_input, d_output)


def dcgan_discriminator(bnmode=0):
    global usedbn, conv_init

    d_input = Input(shape=dim_ordering_shape((3, 32, 32)))

    L = ZeroPadding2D()(d_input)
    L = Conv2D(filters=64, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if(usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = LeakyReLU(alpha=0.2)(L)

    L = ZeroPadding2D()(L)
    L = Conv2D(filters=128, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = LeakyReLU(alpha=0.2)(L)

    L = ZeroPadding2D()(L)
    L = Conv2D(filters=256, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = LeakyReLU(alpha=0.2)(L)

    L = Conv2D(filters=1, kernel_size=4, strides=1, use_bias=False, kernel_initializer=conv_init)(L)
    L = Flatten()(L)
    d_output = Activation('sigmoid')(L)

    return Model(d_input, d_output)


def fccgan_generator(bnmode=0, fc_layers=3):
    global usegbn, conv_init

    #FC layers
    d_input = Input(shape=(100,))
    L = d_input

    if(fc_layers>3):
        L = Dense(16)(L)
        L = Activation('relu')(L)

    if(fc_layers>1):
        L = Dense(64)(L)
        L = Activation('relu')(L)

    if(fc_layers>2):
        L = Dense(512)(L)
        L = Activation('relu')(L)

    if(fc_layers>4):
        L = Dense(512)(L)
        L = Activation('relu')(L)

    if(fc_layers>5):
        L = Dense(2048)(L)
        L = Activation('relu')(L)

    L = Dense(256 * 4 * 4)(L)
    L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Reshape(dim_ordering_shape((256, 4, 4)))(L)

    L = Conv2DTranspose(filters=128, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=64, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=3, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    d_output = Activation('tanh')(L)

    return Model(d_input, d_output)


def fccgan_discriminator(bnmode=0, fc_layers=4):
    global usedbn, conv_init

    d_input = Input(shape=dim_ordering_shape((3, 32, 32)))

    L = ZeroPadding2D()(d_input)
    L = Conv2D(filters=64, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = LeakyReLU(alpha=0.2)(L)

    L = ZeroPadding2D()(L)
    L = Conv2D(filters=128, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = LeakyReLU(alpha=0.2)(L)

    L = ZeroPadding2D()(L)
    L = Conv2D(filters=256, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = LeakyReLU(alpha=0.2)(L)

    L = Flatten()(L)

    if(fc_layers>4):
        L = Dense(1024)(L)
        L = LeakyReLU(alpha=0.2)(L)

    if(fc_layers>1):
        L = Dense(512)(L)
        L = LeakyReLU(alpha=0.2)(L)

    if(fc_layers>2):
        L = Dense(64)(L)
        L = LeakyReLU(alpha=0.2)(L)

    if(fc_layers>3):
        L = Dense(16)(L)
        L = LeakyReLU(alpha=0.2)(L)

    if(fc_layers>5):
        L = Dense(8)(L)
        L = LeakyReLU(alpha=0.2)(L)

    L = Dense(1)(L)
    d_output = Activation('sigmoid')(L)

    return Model(d_input, d_output)


def fccgan_discriminator_pooling(bnmode=0, fc_layers=4):
    global usedbn, conv_init

    d_input = Input(shape=dim_ordering_shape((3, 32, 32)))

    L = Conv2D(filters=64, kernel_size=4, strides=1, padding='same', use_bias=False, kernel_initializer=conv_init)(d_input)
    # if(usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = AveragePooling2D(pool_size=(2, 2))(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Conv2D(filters=128, kernel_size=4, strides=1, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = AveragePooling2D(pool_size=(2, 2))(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Conv2D(filters=256, kernel_size=4, strides=1, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = AveragePooling2D(pool_size=(2, 2))(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Flatten()(L)

    if(fc_layers>4):
        L = Dense(1024)(L)
        L = LeakyReLU(alpha=0.2)(L)

    if(fc_layers>1):
        L = Dense(512)(L)
        L = LeakyReLU(alpha=0.2)(L)

    if(fc_layers>2):
        L = Dense(64)(L)
        L = LeakyReLU(alpha=0.2)(L)

    if(fc_layers>3):
        L = Dense(16)(L)
        L = LeakyReLU(alpha=0.2)(L)

    if(fc_layers>5):
        L = Dense(8)(L)
        L = LeakyReLU(alpha=0.2)(L)

    L = Dense(1)(L)
    d_output = Activation('sigmoid')(L)

    return Model(d_input, d_output)



def dcgan_generator_large(bnmode=0, act='relu'):
    global usegbn, conv_init

    d_input = Input(shape=(100,))

    L = Reshape(target_shape=dim_ordering_shape((100, 1, 1)))(d_input)

    L = Conv2DTranspose(filters=256, kernel_size=4, strides=1, use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2DTranspose(filters=128, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2DTranspose(filters=64, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2D(filters=64, kernel_size=3, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2DTranspose(filters=3, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    d_output = Activation('tanh')(L)

    return Model(d_input, d_output)

def dcgan_discriminator_large(bnmode=0, act='leakyrelu'):
    global usedbn, conv_init

    d_input = Input(shape=dim_ordering_shape((3, 32, 32)))

    L = Conv2D(filters=64, kernel_size=3, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(d_input)
    if(usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = ZeroPadding2D()(L)
    L = Conv2D(filters=64, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if(usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(L)
    if(usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = ZeroPadding2D()(L)
    L = Conv2D(filters=128, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(L)
    if(usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = ZeroPadding2D()(L)
    L = Conv2D(filters=256, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2D(filters=1, kernel_size=4, strides=1, use_bias=False, kernel_initializer=conv_init)(L)
    L = Flatten()(L)
    d_output = Activation('sigmoid')(L)

    return Model(d_input, d_output)


def fccgan_generator_large(bnmode=0, act='leakyrelu'):
    global usegbn, conv_init

    #LID improvement dense layers
    d_input = Input(shape=(100,))
    L = d_input

    L = Dense(64)(L)
    L = activation(act, L)

    L = Dense(512)(L)
    L = activation(act, L)

    L = Dense(256 * 4 * 4)(L)
    L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Reshape(dim_ordering_shape((256, 4, 4)))(L)

    L = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2DTranspose(filters=128, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2DTranspose(filters=64, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2D(filters=64, kernel_size=3, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2DTranspose(filters=3, kernel_size=4, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    L = Cropping2D(cropping=1)(L)
    d_output = Activation('tanh')(L)

    return Model(d_input, d_output)


def fccgan_discriminator_large_pooling(bnmode=0, act='leakyrelu'):
    global usedbn, conv_init

    d_input = Input(shape=dim_ordering_shape((3, 32, 32)))

    L = Conv2D(filters=64, kernel_size=3, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(d_input)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2D(filters=64, kernel_size=4, strides=1, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = AveragePooling2D(pool_size=(2, 2))(L)
    L = activation(act, L)

    L = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2D(filters=128, kernel_size=4, strides=1, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = AveragePooling2D(pool_size=(2, 2))(L)
    L = activation(act, L)

    L = Conv2D(filters=256, kernel_size=3, strides=1, use_bias=False, padding='same', kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = activation(act, L)

    L = Conv2D(filters=256, kernel_size=4, strides=1, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = AveragePooling2D(pool_size=(2, 2))(L)
    L = activation(act, L)

    L = Flatten()(L)

    L = Dense(64)(L)
    L = activation(act, L)

    L = Dense(16)(L)
    L = activation(act, L)

    L = Dense(1)(L)
    d_output = Activation('sigmoid')(L)

    return Model(d_input, d_output)


def run_gan(exp_dir, adversarial_optimizer, opt_g, opt_d, generator, discriminator, latent_dim,
                targets=gan_targets, loss='binary_crossentropy', batch_size = 32, data_size = 50000):
    #print models
    generator.summary()
    discriminator.summary()
    gan = simple_gan(generator=generator,
                     discriminator=discriminator,
                     latent_sampling=normal_latent_sampling((latent_dim,)))

    # build adversarial model
    model = AdversarialModel(base_model=gan, player_params=[generator.trainable_weights, discriminator.trainable_weights],
                             player_names=["generator", "discriminator"])

    model.adversarial_compile(adversarial_optimizer=adversarial_optimizer, player_optimizers=[opt_g, opt_d], loss=loss)

    # create callback to generate images
    zsamples = np.random.normal(size=(10 * 10, latent_dim))
    def generator_sampler():
        xpred = dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1))
        xpred = scale_value(xpred, [0.0, 1.0])
        return xpred.reshape((10, 10) + xpred.shape[1:])

    save_image_cb = ImageGridCallback('./dcgan-v2-images/' + exp_dir + '/epoch-{:03d}.png', generator_sampler, cmap=None)
    save_model_cb = SaveModelWeights(generator, './dcgan-v2-model-weights/' + exp_dir)
  
    # train model
    xtrain, xtest = cifar10_data(data_size)
    y = targets(xtrain.shape[0])
    ytest = targets(xtest.shape[0])
    callbacks = [save_image_cb, save_model_cb]

    #train model
    epoch_start = 0
    epoch_count = 100
    history = fit(model, x=xtrain, y=y, validation_data=(xtest, ytest), callbacks=callbacks, nb_epoch=epoch_start + epoch_count,
                  batch_size=batch_size, initial_epoch = epoch_start, shuffle=True)

    # save history to CSV
    df = pd.DataFrame(history.history)
    df.to_csv('./dcgan-v2-images/' + exp_dir + '/history.csv')

    #save final models
    generator.save('./dcgan-v2-model-weights/' + exp_dir + '/generator.h5')
    discriminator.save('./dcgan-v2-model-weights/' + exp_dir + '/discriminator.h5')


				
if __name__ == "__main__":
    print('')
    latent_dim = 100  # input_dim
    generator = fccgan_generator(bnmode=1)
    discriminator = fccgan_discriminator_pooling(bnmode=1)

    run_gan('fccgan_pooling', AdversarialOptimizerSimultaneous(), opt_g=Adam(0.0001, decay=1e-5), opt_d=Adam(0.0001, decay=1e-5),
                generator=generator, discriminator=discriminator, batch_size=64, latent_dim=latent_dim)


