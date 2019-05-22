from keras_adversarial.legacy import fit
from myutils import scale_value
from keras.initializers import RandomNormal
from keras.callbacks import Callback
from keras_adversarial import gan_targets, AdversarialModel, simple_gan, normal_latent_sampling, \
    AdversarialOptimizerSimultaneous
from keras_adversarial.image_grid_callback import ImageGridCallback

from image_utils import dim_ordering_unfix, dim_ordering_shape
from keras.models import Model
from keras.layers import Conv2D, BatchNormalization, Input, Dense, Dropout, AveragePooling2D
from keras.layers import Conv2DTranspose, Reshape, Activation, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
from sklearn.model_selection import train_test_split
import numpy as np
import pandas as pd
from keras.datasets import mnist

from myutils import get_bn_axis

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

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

    L = Conv2DTranspose(filters=128, kernel_size=3, strides=1, use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=64, kernel_size=3, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    d_output = Activation('tanh')(L)

    return Model(d_input, d_output)


def dcgan_discriminator(bnmode=0):
    global usedbn, conv_init

    d_input = Input(shape=dim_ordering_shape((1, 28, 28)))

    L = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=conv_init)(d_input)
    if(usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = LeakyReLU(alpha=0.2)(L)

    L = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = LeakyReLU(alpha=0.2)(L)

    L = Conv2D(filters=128, kernel_size=3, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = LeakyReLU(alpha=0.2)(L)

    L = Conv2D(filters=1, kernel_size=3, strides=1, use_bias=False, kernel_initializer=conv_init)(L)
    L = Flatten()(L)
    d_output = Activation('sigmoid')(L)

    return Model(d_input, d_output)


def fccgan_generator(bnmode=0, fc_layers = 3):
    global usegbn, conv_init

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

    L = Dense(128 * 3 * 3)(L)
    if (usegbn):  L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Reshape(dim_ordering_shape((128, 3, 3)))(L)

    L = Conv2DTranspose(filters=32, kernel_size=3, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    d_output = Activation('tanh')(L)

    return Model(d_input, d_output)


def fccgan_discriminator(bnmode=0):
    global usedbn, conv_init

    d_input = Input(shape=dim_ordering_shape((1, 28, 28)))

    L = Conv2D(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=conv_init)(d_input)
    if(usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = LeakyReLU(alpha=0.2)(L)

    L = Conv2D(filters=64, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = LeakyReLU(alpha=0.2)(L)

    L = Conv2D(filters=128, kernel_size=3, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = LeakyReLU(alpha=0.2)(L)

    L = Flatten()(L)

    L = Dense(512)(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Dense(64)(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Dense(16)(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Dense(1)(L)
    d_output = Activation('sigmoid')(L)

    return Model(d_input, d_output)


def fccgan_discriminator_pooling(bnmode=0, fc_layers = 4):
    global usedbn, conv_init

    d_input = Input(shape=dim_ordering_shape((1, 28, 28)))

    L = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=conv_init)(d_input)
    if(usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = AveragePooling2D(pool_size=(2,2))(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = AveragePooling2D(pool_size=(2,2))(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = AveragePooling2D(pool_size=(2,2))(L)
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

def mnist_process(x):
    x = np.reshape(x, newshape=(-1, 28, 28, 1))
    x = x.astype(np.float32)
    x = scale_value(x, [-1.0, 1.0]) #rescale to [-1, 1] for compatible with tanh output
    return x


def load_images():
    # returns svhn images, dimension: [32,32,3] value_range: [0, 255]
    train_data = io.loadmat('data/train_32x32.mat')
    images = train_data['X']
    images = np.transpose(images, (3,0,1,2))
    labels = train_data['y']
    labels[labels == 10] = 0
    return images, labels


def mnist_data(data_size):
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()

    if(data_size < xtrain.shape[0]):
        sel_pct =  data_size / xtrain.shape[0]
        print('Selecting percent: ', sel_pct)
        (X_train1, X_train2, Y_train1, Y_train2) = train_test_split(xtrain, ytrain, train_size=sel_pct, stratify=ytrain)
    else:
        X_train1 = xtrain

    return mnist_process(X_train1), mnist_process(xtest)


class SaveModelWeights(Callback):
    def __init__(self, g, path):
        self.generator = g
        self.save_path = path

    def on_epoch_end(self, epoch, logs={}):
        save_file_name = self.save_path + '/gen_weight_epoch_' + str(epoch) + '.h5'
        self.generator.save_weights(save_file_name)


def run_gan(exp_dir, adversarial_optimizer, opt_g, opt_d, generator, discriminator, latent_dim,
                targets=gan_targets, loss='binary_crossentropy', data_size = 60000):
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
        images = dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1))
        images = scale_value(images, [0.0, 1.0]) #rescale tanh output to [0, 1] for display
        return images.reshape((10, 10, 28, 28))

    save_image_cb = ImageGridCallback('./dcgan-v2-images/' + exp_dir + '/epoch-{:03d}.png', generator_sampler, cmap='gray')
    save_model_cb = SaveModelWeights(generator, './dcgan-v2-model-weights/' + exp_dir)

    # train model
    xtrain, xtest = mnist_data(data_size) #set shuffle = True
    shuffle = True

    y = targets(xtrain.shape[0])
    ytest = targets(xtest.shape[0])
    callbacks = [save_image_cb]



    #train model
    epoch_start = 0
    epoch_count = 50
    history = fit(model, x=xtrain, y=y, validation_data=(xtest, ytest), callbacks=callbacks, nb_epoch=epoch_start + epoch_count,
                  batch_size=32, initial_epoch = epoch_start, shuffle=shuffle)

    # save history to CSV
    df = pd.DataFrame(history.history)
    df.to_csv('./dcgan-v2-images/' + exp_dir + '/history.csv')

    #save final models
    generator.save('./dcgan-v2-model-weights/' + exp_dir + '/generator.h5')
    discriminator.save('./dcgan-v2-model-weights/' + exp_dir + '/discriminator.h5')





if __name__ == "__main__":
    latent_dim = 100 #input_dim

    generator = fccgan_generator(bnmode=1)
    discriminator = fccgan_discriminator_pooling(bnmode=1)
    run_gan('fccgan_pooling', AdversarialOptimizerSimultaneous(), opt_g=Adam(0.0002, decay=1e-5), opt_d=Adam(0.0001, decay=1e-5),
                generator=generator, discriminator=discriminator, latent_dim=latent_dim)


