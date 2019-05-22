from keras.initializers import RandomNormal
from keras.callbacks import Callback
from keras_adversarial import gan_targets, AdversarialModel, simple_gan, normal_latent_sampling, \
    AdversarialOptimizerSimultaneous
from keras_adversarial.image_grid_callback import ImageGridCallback
from keras_adversarial.legacy import fit, Convolution2D
from image_utils import dim_ordering_unfix, dim_ordering_shape
from keras.models import Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dense, AveragePooling2D
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.optimizers import Adam
import numpy as np
from myutils import get_bn_axis, scale_value
import pandas as pd
from scipy import io

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)


def load_images():
    # returns svhn images, dimension: [32,32,3] value_range: [0, 255]
    train_data = io.loadmat('data/train_32x32.mat')
    images = train_data['X']
    images = np.transpose(images, (3,0,1,2))
    labels = train_data['y']
    labels[labels == 10] = 0
    return images, labels

def svhn_process(x):
    x = x.astype(np.float32)
    x = scale_value(x, [-1, 1]) #rescale to tanh compatible output
    return x

def svhn_data():
    xtr1, ytr1 = load_images()
    test_pct = 0.20
    test_size = int(np.ceil(xtr1.shape[0]*test_pct))
    xtr2 = xtr1[0: test_size]
    # (xtr1, xtr2, ytr1, ytr2) = train_test_split(xtr1, ytr1, train_size=train_pct, stratify=ytr1)
    return svhn_process(xtr1), svhn_process(xtr2)


class SaveModelWeights(Callback):
    def __init__(self, g, path):
        self.generator = g
        self.save_path = path

    def on_epoch_end(self, epoch, logs={}):
        save_file_name = self.save_path + '/gen_weight_epoch_' + str(epoch) + '.h5'
        self.generator.save_weights(save_file_name)

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


def fccgan_generator(bnmode=0):
    global usegbn, conv_init

    #LID improvement dense layers
    d_input = Input(shape=(100,))

    L = Dense(64)(d_input)
    L = Activation('relu')(L)

    L = Dense(512)(L)
    L = Activation('relu')(L)

    L = Dense(256 * 4 * 4)(L)
    if (usegbn):  L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
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


def fccgan_discriminator(bnmode=0):
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

    L = Dense(512)(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Dense(64)(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Dense(16)(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Dense(1)(L)
    d_output = Activation('sigmoid')(L)

    return Model(d_input, d_output)


def fccgan_discriminator_pooling(bnmode=0):
    global usedbn, conv_init

    d_input = Input(shape=dim_ordering_shape((3, 32, 32)))

    L = Conv2D(filters=64, kernel_size=4, strides=1, padding='same', use_bias=False, kernel_initializer=conv_init)(d_input)
    if(usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
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

    L = Dense(512)(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Dense(64)(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Dense(16)(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Dense(1)(L)
    d_output = Activation('sigmoid')(L)

    return Model(d_input, d_output)



def run_gan(exp_dir, adversarial_optimizer, opt_g, opt_d, generator, discriminator, latent_dim,
                targets=gan_targets, loss='binary_crossentropy'):
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
    xtrain, xtest = svhn_data()
    y = targets(xtrain.shape[0])
    ytest = targets(xtest.shape[0])
    callbacks = [save_image_cb, save_model_cb]

    #train model
    epoch_start = 0
    epoch_count = 100
    history = fit(model, x=xtrain, y=y, validation_data=(xtest, ytest), callbacks=callbacks, nb_epoch=epoch_start + epoch_count,
                  batch_size=32, initial_epoch = epoch_start, shuffle=True)

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
    run_gan('fccgan_pooling', AdversarialOptimizerSimultaneous(), opt_g=Adam(0.0001, decay=1e-5), opt_d=Adam(0.0001, decay=1e-5),
                generator=generator, discriminator=discriminator, latent_dim=latent_dim)