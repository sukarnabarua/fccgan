import keras.backend as K
from keras_adversarial.image_grid import write_image_grid
from image_utils import dim_ordering_unfix, dim_ordering_shape
from keras.models import Model
from keras.layers import Conv2D, ZeroPadding2D, BatchNormalization, Input, Dense, UpSampling2D, MaxPooling2D, \
    AveragePooling2D
from keras.layers import Conv2DTranspose, Reshape, Activation, Cropping2D, Flatten
from keras.layers.advanced_activations import LeakyReLU
from keras.initializers import RandomNormal
from keras.optimizers import RMSprop
from keras.datasets import cifar10
import numpy as np
import pandas as pd

from myutils import get_bn_axis, scale_value

conv_init = RandomNormal(0, 0.02)
gamma_init = RandomNormal(1., 0.02)

def cifar10_process(x):
    x = x.astype(np.float32)
    x = scale_value(x, [-1, 1]) #rescale to tanh compatible output
    return x

def cifar10_data():
    (xtrain, ytrain), (xtest, ytest) = cifar10.load_data()
    return cifar10_process(xtrain), cifar10_process(xtest)

def generate_images(generator, image_path, epoch, cmap='gray'):
    xsamples = generator()
    xsamples = scale_value(xsamples, [0, 1])  # convert tanh output to [0, 1] for display
    image_path = image_path.format(epoch)
    write_image_grid(image_path, xsamples, cmap=cmap)

def generator_sampler(latent_dim, generator):
    def fun():
        zsamples = np.random.normal(size=(10 * 10, latent_dim))
        xpred = dim_ordering_unfix(generator.predict(zsamples)).transpose((0, 2, 3, 1))
        return xpred.reshape((10, 10) + xpred.shape[1:])
    return fun


usegbn = True
usedbn = True

def setbn(gbn = True, dbn = True):
    global usegbn
    global usedbn
    usegbn = gbn
    usedbn = dbn


def wgan_generator(bnmode=0):
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

def wgan_discriminator(bnmode=0):
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
    d_output = Flatten()(L)

    return Model(d_input, d_output)


def fcc_wgan_generator(bnmode=0, fc_layers = 3):
    global usegbn, conv_init

    nch = 256
    #LID improvement dense layers
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

    L = Dense(nch * 4 * 4)(L)
    if (usegbn):  L = BatchNormalization(axis=get_bn_axis(), epsilon=1.01e-5)(L, training=bnmode)
    L = Reshape(dim_ordering_shape((nch, 4, 4)))(L)

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


def fcc_wgan_discriminator(bnmode=0):
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

    d_output = Dense(1)(L)

    return Model(d_input, d_output)


def fcc_wgan_discriminator_uc(bnmode=0, fc_layers=4):
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

    if (fc_layers>4):
        L = Dense(1024)(L)
        L = LeakyReLU(alpha=0.2)(L)

    if (fc_layers>3):
        L = Dense(256)(L)
        L = LeakyReLU(alpha=0.2)(L)

    if (fc_layers>2):
        L = Dense(64)(L)
        L = LeakyReLU(alpha=0.2)(L)

    if (fc_layers>1):
        L = Dense(16)(L)
        L = LeakyReLU(alpha=0.2)(L)

    d_output = Dense(1)(L)

    return Model(d_input, d_output)


def run_wgan_exp(exp_dir, netG, netD):

    K.set_image_data_format('channels_last')

    netG.summary()
    netD.summary()

    nz = 100 #noise dimension
    ngf = 64
    ndf = 64
    n_extra_layers = 0
    Diters = 5

    imageSize = 32 #image height and width
    nc = 3 #image channels

    batchSize = 64
    lrD = 0.00005
    lrG = 0.00005

    #define weight clipping function
    clamp_lower, clamp_upper = -0.01, 0.01 #wgan weight clip maximum and minimum value
    clamp_updates = [K.update(v, K.clip(v, clamp_lower, clamp_upper)) for v in netD.trainable_weights]
    netD_clamp = K.function([], [], clamp_updates)

    #define wgan loss discriminator
    netD_real_input = Input(shape=dim_ordering_shape((nc, imageSize, imageSize)) )
    noisev = Input(shape=(nz,))
    loss_real = K.mean(netD(netD_real_input))
    loss_fake = K.mean(netD(netG(noisev)))
    loss = loss_fake - loss_real
    training_updates = RMSprop(lr=lrD).get_updates(netD.trainable_weights, [], loss)
    netD_train = K.function([netD_real_input, noisev], [loss_real, loss_fake], training_updates)

    #define wgan loss for generator
    loss = -loss_fake
    training_updates = RMSprop(lr=lrG).get_updates(netG.trainable_weights,[], loss)
    netG_train = K.function([noisev], [loss], training_updates)

    #get real data and noise
    train_X, test_X = cifar10_data()
    start = 0
    niter = 100
    gen_iterations = 0
    loss = []
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
        print('Discriminator loss: ', 100 * (errD_fake - errD_real), 'Generator loss: ', 100 * errG)
        loss.extend([errD_fake, errD_real, errG])

        # rem = epoch % 100
        # if rem <= 10:
        generate_images(generator_sampler(nz, netG), './wgan-v2-images/' + exp_dir + '/epoch-{:03d}.png', epoch)
        netG.save_weights('./wgan-v2-model-weights/' + exp_dir + '/gen_weight_epoch_' + str(epoch) + '.h5')


    #save loss history
    df = pd.DataFrame(loss)
    df.to_csv('./wgan-v2-images/' + exp_dir + '/history.csv')

    #save final weights
    netG.save_weights('./wgan-v2-model-weights/' + exp_dir + '/generator.h5')
    netD.save_weights('./wgan-v2-model-weights/' + exp_dir + '/discriminator.h5')


if __name__ == '__main__':
    print('')

    # create models
    netG = fcc_wgan_generator(bnmode=1)
    netD = fcc_wgan_discriminator_uc(bnmode=1)
    run_wgan_exp('fccgan_pooling', netG, netD)
