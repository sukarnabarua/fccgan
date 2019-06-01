import os, sys
from myutils import scale_value
sys.path.append(os.getcwd())

import tflib as lib
import tflib.ops.linear
import tflib.ops.cond_batchnorm
import tflib.ops.conv2d
import tflib.ops.batchnorm
import tflib.save_images
import tflib.cifar10
from inception import get_inception_score

import numpy as np
import tensorflow as tf
import functools
import locale
locale.setlocale(locale.LC_ALL, '')

# Download CIFAR-10 (Python version) at
# https://www.cs.toronto.edu/~kriz/cifar.html and fill in the path to the
# extracted files here!
DATA_DIR = 'cifar10data'
if len(DATA_DIR) == 0:
    raise Exception('Please specify path to data directory in gan_cifar.py!')

N_GPUS = 1
if N_GPUS not in [1,2]:
    raise Exception('Only 1 or 2 GPUs supported!')

BATCH_SIZE = 64 # Critic batch size
GEN_BS_MULTIPLE = 2 # Generator batch size, as a multiple of BATCH_SIZE
ITERS = 100000 # How many iterations to train for
DIM_G = 128 # Generator dimensionality
DIM_D = 128 # Critic dimensionality
NORMALIZATION_G = True # Use batchnorm in generator?
NORMALIZATION_D = False # Use batchnorm (or layernorm) in critic?
OUTPUT_DIM = 3072 # Number of pixels in CIFAR10 (32*32*3)
LR = 2e-4 # Initial learning rate
DECAY = True # Whether to decay LR over learning
N_CRITIC = 5 # Critic steps per generator steps
INCEPTION_FREQUENCY = 1000 # How frequently to calculate Inception score

CONDITIONAL = False # Whether to train a conditional or unconditional model
ACGAN = False # If CONDITIONAL, whether to use ACGAN or "vanilla" conditioning
ACGAN_SCALE = 1. # How to scale the critic's ACGAN loss relative to WGAN loss
ACGAN_SCALE_G = 0.1 # How to scale generator's ACGAN loss relative to WGAN loss

if CONDITIONAL and (not ACGAN) and (not NORMALIZATION_D):
    print("WARNING! Conditional model without normalization in D might be effectively unconditional!")

DEVICES = ['/gpu:{}'.format(i) for i in range(N_GPUS)]
if len(DEVICES) == 1: # Hack because the code assumes 2 GPUs
    DEVICES = [DEVICES[0], DEVICES[0]]

lib.print_model_settings(locals().copy())

def nonlinearity(x):
    # return tf.nn.relu(x)
    return tf.nn.leaky_relu(x)

def Normalize(name, inputs,labels=None):
    """This is messy, but basically it chooses between batchnorm, layernorm,
    their conditional variants, or nothing, depending on the value of `name` and
    the global hyperparam flags."""
    if not CONDITIONAL:
        labels = None
    if CONDITIONAL and ACGAN and ('Discriminator' in name):
        labels = None

    if ('Discriminator' in name) and NORMALIZATION_D:
        return lib.ops.layernorm.Layernorm(name,[1,2,3],inputs,labels=labels,n_labels=10)
    elif ('Generator' in name) and NORMALIZATION_G:
        if labels is not None:
            return lib.ops.cond_batchnorm.Batchnorm(name,[0,2,3],inputs,labels=labels,n_labels=10)
        else:
            return lib.ops.batchnorm.Batchnorm(name,[0,2,3],inputs,fused=True)
    else:
        return inputs

def ConvMeanPool(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, inputs, he_init=he_init, biases=biases)
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    return output

def MeanPoolConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.add_n([output[:,:,::2,::2], output[:,:,1::2,::2], output[:,:,::2,1::2], output[:,:,1::2,1::2]]) / 4.
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def UpsampleConv(name, input_dim, output_dim, filter_size, inputs, he_init=True, biases=True):
    output = inputs
    output = tf.concat([output, output, output, output], axis=1)
    output = tf.transpose(output, [0,2,3,1])
    output = tf.depth_to_space(output, 2)
    output = tf.transpose(output, [0,3,1,2])
    output = lib.ops.conv2d.Conv2D(name, input_dim, output_dim, filter_size, output, he_init=he_init, biases=biases)
    return output

def ResidualBlock(name, input_dim, output_dim, filter_size, inputs, resample=None, no_dropout=False, labels=None):
    """
    resample: None, 'down', or 'up'
    """
    if resample=='down':
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=input_dim)
        conv_2        = functools.partial(ConvMeanPool, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = ConvMeanPool
    elif resample=='up':
        conv_1        = functools.partial(UpsampleConv, input_dim=input_dim, output_dim=output_dim)
        conv_shortcut = UpsampleConv
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    elif resample==None:
        conv_shortcut = lib.ops.conv2d.Conv2D
        conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=input_dim, output_dim=output_dim)
        conv_2        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=output_dim, output_dim=output_dim)
    else:
        raise Exception('invalid resample value')

    if output_dim==input_dim and resample==None:
        shortcut = inputs # Identity skip-connection
    else:
        shortcut = conv_shortcut(name+'.Shortcut', input_dim=input_dim, output_dim=output_dim, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = Normalize(name+'.N1', output, labels=labels)
    output = nonlinearity(output)
    output = conv_1(name+'.Conv1', filter_size=filter_size, inputs=output)
    output = Normalize(name+'.N2', output, labels=labels)
    output = nonlinearity(output)
    output = conv_2(name+'.Conv2', filter_size=filter_size, inputs=output)

    return shortcut + output


def OptimizedResBlockDisc1(inputs):
    conv_1        = functools.partial(lib.ops.conv2d.Conv2D, input_dim=3, output_dim=DIM_D)
    conv_2        = functools.partial(ConvMeanPool, input_dim=DIM_D, output_dim=DIM_D)
    conv_shortcut = MeanPoolConv
    shortcut = conv_shortcut('Discriminator.1.Shortcut', input_dim=3, output_dim=DIM_D, filter_size=1, he_init=False, biases=True, inputs=inputs)

    output = inputs
    output = conv_1('Discriminator.1.Conv1', filter_size=3, inputs=output)
    output = nonlinearity(output)
    output = conv_2('Discriminator.1.Conv2', filter_size=3, inputs=output)
    return shortcut + output


def Generator(n_samples, labels, noise=None):
    if noise is None:
        noise = tf.random_normal([int(n_samples), 128])

    #FCC-GAN FC layers
    output = lib.ops.linear.Linear('Generator.FC1', 128, 64, noise)
    output = nonlinearity(output)

    output = lib.ops.linear.Linear('Generator.FC2', 64, 512, output)
    output = nonlinearity(output)

    output = lib.ops.linear.Linear('Generator.Input', 512, 4*4*DIM_G, output)
    output = tf.reshape(output, [-1, DIM_G, 4, 4])

    #Residual blocks
    output = ResidualBlock('Generator.1', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.2', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = ResidualBlock('Generator.3', DIM_G, DIM_G, 3, output, resample='up', labels=labels)
    output = Normalize('Generator.OutputN', output)
    output = nonlinearity(output)
    output = lib.ops.conv2d.Conv2D('Generator.Output', DIM_G, 3, 3, output, he_init=False)
    output = tf.tanh(output)
    return tf.reshape(output, [-1, OUTPUT_DIM])



def Discriminator(inputs, labels):
    output = tf.reshape(inputs, [-1, 3, 32, 32])
    output = OptimizedResBlockDisc1(output)
    output = ResidualBlock('Discriminator.2', DIM_D, DIM_D, 3, output, resample='down', labels=labels)
    output = ResidualBlock('Discriminator.3', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = ResidualBlock('Discriminator.4', DIM_D, DIM_D, 3, output, resample=None, labels=labels)
    output = nonlinearity(output)
    output = tf.reduce_mean(output, axis=[2,3])
    features = output

    #FCC-GAN FC layers
    output = lib.ops.linear.Linear('Discriminator.FC3', 128, 16, output)
    output = nonlinearity(output)

    output = lib.ops.linear.Linear('Discriminator.Output', 16, 1, output)
    output_wgan = tf.reshape(output, [-1])

    if CONDITIONAL and ACGAN:
        output_acgan = lib.ops.linear.Linear('Discriminator.ACGANOutput', DIM_D, 10, output)
        return output_wgan, output_acgan, features
    else:
        return output_wgan, None, features


def get_stdgan_loss_dis(real_outputs, fake_outputs, batch_size = 64):
    real_outputs = tf.reshape(real_outputs, (-1, 1))
    fake_outputs = tf.reshape(fake_outputs, (-1, 1))
    real_targets = tf.constant(np.ones((batch_size, 1)).astype(np.float32), shape=(batch_size, 1))
    fake_targets = tf.constant(np.zeros((batch_size, 1)).astype(np.float32), shape=(batch_size, 1))
    loss_real = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_targets,logits=real_outputs))
    loss_fake = tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=fake_targets,logits=fake_outputs))
    return 5.0 * (loss_real + loss_fake)
    #a multiplication factor for loss can balance between generator and discriminator update rate


def get_stdgan_loss_gen(fake_outputs, batch_size = 64):
    fake_outputs = tf.reshape(fake_outputs, (-1, 1))
    real_targets = tf.constant(np.ones((batch_size, 1)).astype(np.float32), shape=(batch_size, 1))
    return 2.0 * tf.reduce_mean(tf.nn.sigmoid_cross_entropy_with_logits(labels=real_targets,logits=fake_outputs))
    # a multiplication factor for loss can balance between generator and discriminator update rate

#Set to False to train, True to load pretrained model and compute Inception score
pretrained = False

fake_labels_100 = tf.cast(tf.random_uniform([100]) * 10, tf.int32)
samples_100 = Generator(100, fake_labels_100)

if not pretrained:
    print('Training model')
    with tf.Session() as session:
        _iteration = tf.placeholder(tf.int32, shape=None)
        all_real_data_int = tf.placeholder(tf.int32, shape=[BATCH_SIZE, OUTPUT_DIM])
        all_real_labels = tf.placeholder(tf.int32, shape=[BATCH_SIZE])

        labels_splits = tf.split(all_real_labels, len(DEVICES), axis=0)

        fake_data_splits = []
        for i, device in enumerate(DEVICES):
            with tf.device(device):
                fake_data_splits.append(Generator(BATCH_SIZE/len(DEVICES), labels_splits[i]))

        all_real_data = tf.reshape(2*((tf.cast(all_real_data_int, tf.float32)/256.)-.5), [BATCH_SIZE, OUTPUT_DIM])
        all_real_data += tf.random_uniform(shape=[BATCH_SIZE,OUTPUT_DIM],minval=0.,maxval=1./128) # dequantize
        all_real_data_splits = tf.split(all_real_data, len(DEVICES), axis=0)

        DEVICES_B = DEVICES[:int(len(DEVICES)/2)]
        DEVICES_A = DEVICES[int(len(DEVICES)/2):]

        disc_costs = []
        disc_acgan_costs = []
        disc_acgan_accs = []
        disc_acgan_fake_accs = []
        for i, device in enumerate(DEVICES_A):
            with tf.device(device):
                real_and_fake_data = tf.concat([
                    all_real_data_splits[i],
                    all_real_data_splits[len(DEVICES_A)+i],
                    fake_data_splits[i],
                    fake_data_splits[len(DEVICES_A)+i]
                ], axis=0)
                real_and_fake_labels = tf.concat([
                    labels_splits[i],
                    labels_splits[len(DEVICES_A)+i],
                    labels_splits[i],
                    labels_splits[len(DEVICES_A)+i]
                ], axis=0)
                disc_all, disc_all_acgan, disc_all_features = Discriminator(real_and_fake_data, real_and_fake_labels)
                disc_real = disc_all[:int(BATCH_SIZE/len(DEVICES_A))]
                disc_fake = disc_all[int(BATCH_SIZE/len(DEVICES_A)):]
                disc_features_real = disc_all_features[:int(BATCH_SIZE/len(DEVICES_A))]
                disc_features_fake = disc_all_features[int(BATCH_SIZE/len(DEVICES_A)):]
                stdgan_loss_dis = get_stdgan_loss_dis(disc_real, disc_fake, batch_size=BATCH_SIZE)
                disc_loss = stdgan_loss_dis
                disc_costs.append(disc_loss)

        for i, device in enumerate(DEVICES_B):
            with tf.device(device):
                real_data = tf.concat([all_real_data_splits[i], all_real_data_splits[len(DEVICES_A)+i]], axis=0)
                fake_data = tf.concat([fake_data_splits[i], fake_data_splits[len(DEVICES_A)+i]], axis=0)
                labels = tf.concat([
                    labels_splits[i],
                    labels_splits[len(DEVICES_A)+i],
                ], axis=0)
                alpha = tf.random_uniform(
                    shape=[int(BATCH_SIZE/len(DEVICES_A)),1],
                    minval=0.,
                    maxval=1.
                )
                differences = fake_data - real_data
                interpolates = real_data + (alpha*differences)
                gradients = tf.gradients(Discriminator(interpolates, labels)[0], [interpolates])[0]
                slopes = tf.sqrt(tf.reduce_sum(tf.square(gradients), reduction_indices=[1]))
                gradient_penalty = 10 * tf.reduce_mean((slopes - 1.) ** 2)
                disc_costs.append(gradient_penalty)

        disc_wgan = tf.add_n(disc_costs) / len(DEVICES_A)

        disc_acgan = tf.constant(0.)
        disc_acgan_acc = tf.constant(0.)
        disc_acgan_fake_acc = tf.constant(0.)
        disc_cost = disc_wgan

        disc_params = lib.params_with_name('Discriminator.')

        if DECAY:
            decay = tf.maximum(0.25, 1.-(tf.cast(_iteration, tf.float32)/ITERS))
        else:
            decay = 1.

        gen_costs = []
        gen_acgan_costs = []
        for i, device in enumerate(DEVICES_A):
            with tf.device(device):
                real_and_fake_data = tf.concat([
                    all_real_data_splits[i],
                    all_real_data_splits[len(DEVICES_A)+i],
                    fake_data_splits[i],
                    fake_data_splits[len(DEVICES_A)+i]
                ], axis=0)
                real_and_fake_labels = tf.concat([
                    labels_splits[i],
                    labels_splits[len(DEVICES_A)+i],
                    labels_splits[i],
                    labels_splits[len(DEVICES_A)+i]
                ], axis=0)

                disc_all, disc_all_acgan, disc_all_features = Discriminator(real_and_fake_data, real_and_fake_labels)
                disc_real = disc_all[:int(BATCH_SIZE/len(DEVICES_A))]
                disc_fake = disc_all[int(BATCH_SIZE/len(DEVICES_A)):]
                disc_features_real = disc_all_features[:int(BATCH_SIZE/len(DEVICES_A))]
                disc_features_fake = disc_all_features[int(BATCH_SIZE/len(DEVICES_A)):]
                stdgan_loss_gen = get_stdgan_loss_gen(disc_fake, batch_size=BATCH_SIZE)
                gen_loss = stdgan_loss_gen
                gen_costs.append(gen_loss)

        gen_cost = (tf.add_n(gen_costs) / len(DEVICES))

        gen_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
        disc_opt = tf.train.AdamOptimizer(learning_rate=LR*decay, beta1=0., beta2=0.9)
        gen_gv = gen_opt.compute_gradients(gen_cost, var_list=lib.params_with_name('Generator'))
        disc_gv = disc_opt.compute_gradients(disc_cost, var_list=disc_params)
        gen_train_op = gen_opt.apply_gradients(gen_gv)
        disc_train_op = disc_opt.apply_gradients(disc_gv)

        # Function for generating samples
        frame_i = [0]
        fixed_noise = tf.constant(np.random.normal(size=(100, 128)).astype('float32'))
        fixed_labels = tf.constant(np.array([0,1,2,3,4,5,6,7,8,9]*10,dtype='int32'))
        fixed_noise_samples = Generator(100, fixed_labels, noise=fixed_noise)
        def generate_image(frame, true_dist):
            samples = session.run(fixed_noise_samples)
            print("Samples: ", samples.shape)
            samples = ((samples+1.)*(255./2)).astype('int32')
            lib.save_images.save_images(samples.reshape((100, 3, 32, 32)), 'cifarresnet/wgangp/samples_{}.png'.format(frame))

        train_gen, dev_gen = lib.cifar10.load(BATCH_SIZE, DATA_DIR)
        def inf_train_gen():
            while True:
                for images,_labels in train_gen():
                    yield images,_labels


        for name,grads_and_vars in [('G', gen_gv), ('D', disc_gv)]:
            print("{} Params:".format(name))
            total_param_count = 0
            for g, v in grads_and_vars:
                shape = v.get_shape()
                shape_str = ",".join([str(x) for x in v.get_shape()])

                param_count = 1
                for dim in shape:
                    param_count *= int(dim)
                total_param_count += param_count

                if g == None:
                    print("\t{} ({}) [no grad!]".format(v.name, shape_str))
                else:
                    print("\t{} ({})".format(v.name, shape_str))
            print("Total param count: {}".format(
                locale.format("%d", total_param_count, grouping=True)
            ))


        def compute_inception_score(n):
            # Function for calculating inception score
            all_samples = []
            for i in range(int(n / 100)):
                all_samples.append(session.run(samples_100))
            all_samples = np.concatenate(all_samples, axis=0)
            all_samples = all_samples.reshape((-1, 3, 32, 32))
            all_samples = scale_value(all_samples, [-1.0, 1.0])
            print(all_samples.shape)
            return get_inception_score(all_samples)

        session.run(tf.initialize_all_variables())

        saver = tf.train.Saver()

        gen = inf_train_gen()

        iters_epoch = int(np.ceil(50000/(BATCH_SIZE * N_CRITIC)))
        epoch = 0
        best = 0
        for iteration in range(ITERS):
            _data = []
            _labels = []
            for i in range(N_CRITIC):
                _data, _labels = next(gen)
                _stdgan_loss_dis, _disc_cost, _ = session.run([stdgan_loss_dis, disc_cost, disc_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})

            _stdgan_loss_gen, _gen_cost, _ = session.run([stdgan_loss_gen, gen_cost, gen_train_op], feed_dict={all_real_data_int: _data, all_real_labels:_labels, _iteration:iteration})

            if (iteration % iters_epoch == (iters_epoch-1)): #epoch completed
                epoch = int(iteration/iters_epoch) #update epoch number
                print("Completed epoch: ", epoch)

            if (iteration % iters_epoch == (iters_epoch-1)):
                print('disc: ', _stdgan_loss_dis, _disc_cost, ' gen_cost: ', _stdgan_loss_gen, _gen_cost)

            if (iteration % iters_epoch == (iters_epoch-1)): #epoch completed
                if (epoch+1)%50<=10 or epoch%5==0:
                    (mean, std) = compute_inception_score(50000)
                    if mean > best:
                        best = mean
                        saver.save(session, './resnet_tf_models/model_fccgan.ckpt')

                    print('Inception score: ', mean, std, ' best: ', best)

            if (iteration % iters_epoch == (iters_epoch-1)):
                generate_image(epoch, _data)

else: #load pretrained model
    print('Loading pretrained model')
    with tf.Session() as session:
        def compute_inception_score(n):
            all_samples = []
            for i in range(int(n / 100)):
                all_samples.append(session.run(samples_100))
            all_samples = np.concatenate(all_samples, axis=0)
            all_samples = all_samples.reshape((-1, 3, 32, 32))
            all_samples = scale_value(all_samples, [-1.0, 1.0])
            print(all_samples.shape)
            return get_inception_score(all_samples)

        saver = tf.train.Saver()
        saver.restore(session, './resnet_tf_models/model_fccgan.ckpt')
        (mean, std) = compute_inception_score(50000)
        print('Inception score: ', mean, std)
