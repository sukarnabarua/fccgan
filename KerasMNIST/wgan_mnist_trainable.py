from keras_adversarial.image_grid import write_image_grid
import keras.backend as K
from keras.datasets import mnist
from image_utils import dim_ordering_unfix, dim_ordering_shape
from keras.models import Sequential, Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dense, AveragePooling2D
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import RMSprop, Adam

import numpy as np
from myutils import get_bn_axis, scale_value
import pandas as pd

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

usegbn = True
usedbn = True


def wgan_generator(bnmode=0):
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

def wgan_discriminator(bnmode=0):
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
    d_output = Flatten()(L)

    return Model(d_input, d_output)


def fcc_wgan_generator(bnmode=0):
    global usegbn, conv_init

    d_input = Input(shape=(100,))

    L = Reshape(target_shape=dim_ordering_shape((100, 1, 1)))(d_input)

    L = Dense(64)(L)
    L = Activation('relu')(L)

    L = Dense(512)(L)
    L = Activation('relu')(L)

    L = Dense(128 * 3 * 3)(L)
    if (usegbn):  L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Reshape(dim_ordering_shape((128, 3, 3)))(L)

    L = Conv2DTranspose(filters=64, kernel_size=3, strides=2, use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=32, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usegbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Activation("relu")(L)

    L = Conv2DTranspose(filters=1, kernel_size=3, strides=2, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    d_output = Activation('tanh')(L)

    return Model(d_input, d_output)

def fcc_wgan_discriminator(bnmode=0):
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

    d_output = Dense(1)(L)

    return Model(d_input, d_output)


def fcc_wgan_discriminator_pooling(bnmode=0):
    global usedbn, conv_init

    d_input = Input(shape=dim_ordering_shape((1, 28, 28)))

    L = Conv2D(filters=32, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=conv_init)(d_input)
    if(usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = AveragePooling2D(pool_size=(2,2))(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Conv2D(filters=64, kernel_size=3, strides=1, padding='same', use_bias=False, kernel_initializer=conv_init)(L)
    if (usedbn): L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = AveragePooling2D(pool_size=(2, 2))(L)
    L = LeakyReLU(alpha=0.2)(L)

    L = Conv2D(filters=128, kernel_size=3, strides=1, use_bias=False, kernel_initializer=conv_init)(L)
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

    d_output = Dense(1)(L)

    return Model(d_input, d_output)


def mnist_process(x):
    x = np.reshape(x, newshape=(-1, 28, 28, 1))
    x = x.astype(np.float32)
    x = scale_value(x, [-1.0, 1.0]) #rescale to [-1, 1] for compatible with tanh output
    return x

def mnist_data():
    (xtrain, ytrain), (xtest, ytest) = mnist.load_data()
    return mnist_process(xtrain), mnist_process(xtest)

def generate_images(generator, image_path, epoch, cmap='gray'):
    xsamples = generator()
    xsamples = scale_value(xsamples, [0, 1]) #convert tanh output to [0, 1] for display
    image_path = image_path.format(epoch)
    write_image_grid(image_path, xsamples, cmap=cmap)


def generator_sampler(latent_dim, generator):
    def fun():
        state = np.random.get_state()
        np.random.seed(0)
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        np.random.set_state(state)
        images = dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1))
        return images.reshape((10, 10, 28, 28))
    return fun


def run_wgan(netG, netD, exp_dir):

    K.set_image_data_format('channels_last')

    nz = 100 #noise dimension
    Diters = 1

    imageSize = 28 #image height and width
    nc = 1 #image channels

    batchSize = 64 #64
    lrD = 0.0001 #0.00005
    lrG = 0.0001 #0.00005
    optG = RMSprop(lr=lrG)
    optD = RMSprop(lr=lrD)

    clamp_lower, clamp_upper = -0.01, 0.01 #wgan weight clip maximum and minimum value
    # clamp_lower, clamp_upper = -100.01, 100.01  # wgan weight clip maximum and minimum value

    #define weight clipping function
    clamp_updates = [K.update(v, K.clip(v, clamp_lower, clamp_upper)) for v in netD.trainable_weights]
    netD_clamp = K.function([], [], clamp_updates)

    #define wgan loss discriminator
    netD_real_input = Input(shape=dim_ordering_shape((nc, imageSize, imageSize)) )
    noisev = Input(shape=(nz,))
    loss_real = K.mean(netD(netD_real_input))
    loss_fake = K.mean(netD(netG(noisev)))
    loss = loss_fake - loss_real
    training_updates = optD.get_updates(netD.trainable_weights, [], loss)
    netD_train = K.function([netD_real_input, noisev], [loss_real, loss_fake], training_updates)

    #define wgan loss for generator
    loss = -loss_fake
    training_updates = optG.get_updates(netG.trainable_weights,[], loss)
    netG_train = K.function([noisev], [loss], training_updates)

    #get real data and noise
    train_X, test_X = mnist_data()
    niter = 100
    start = 0
    gen_iterations = 0
    for epoch in range(start, start + niter):
        i = 0
        np.random.shuffle(train_X)
        batches = train_X.shape[0] // batchSize
        while i < batches:
            _Diters = Diters
            j = 0
            while j < _Diters and i < batches:
                j += 1
                i += 1
                netD_clamp([])
                real_data = train_X[i*batchSize:(i+1)*batchSize]
                noise = np.random.normal(size=(batchSize, nz))
                errD_real, errD_fake  = netD_train([real_data, noise])

            noise = np.random.normal(size=(batchSize, nz))
            errG, = netG_train([noise])

            gen_iterations += 1

        # Save generator model after every epoch
        generate_images(generator_sampler(nz, netG), './wgan-v2-images/' + exp_dir + '/epoch-{:03d}.png', epoch)
        netG.save_weights('./wgan-v2-model-weights/' + exp_dir + '/gen_weight_epoch_' + str(epoch) + '.h5')
        print('Discriminator loss: ', 10000*(errD_fake - errD_real), 'Generator loss: ', 10000*errG)

    #save loss history
    df = pd.DataFrame(loss)
    df.to_csv('./wgan-v2-images/' + exp_dir + '/history.csv')

    #save final weights
    netG.save_weights('./wgan-v2-model-weights/' + exp_dir + '/generator.h5')
    netD.save_weights('./wgan-v2-model-weights/' + exp_dir + '/discriminator.h5')


def generate_images_epochs(exp_dir = 'probexp/0'):
    nz = 100
    netG = wgan_generator(bnmode=1)
    for epoch in range(0, 100):
        netG.load_weights('./dcgan-v2-model-weights/' + exp_dir + '/gen_weight_epoch_' + str(epoch) + '.h5')
        generate_images(generator_sampler(nz, netG), './dcgan-v2-images/' + exp_dir + '/epoch-{:03d}.png', epoch)




if __name__ == '__main__':
    print('')
    generator = fcc_wgan_generator(bnmode=1)
    discriminator = fcc_wgan_discriminator_pooling(bnmode=1)
    run_wgan(generator, discriminator, 'fcc_wgan_pooling')
