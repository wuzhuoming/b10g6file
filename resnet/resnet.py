from __future__ import absolute_import, division, print_function, unicode_literals
import time
import json
import os
from tensorflow.keras.layers import Input, Add, Dense, Activation, ZeroPadding2D, BatchNormalization, Flatten, Conv2D, AveragePooling2D, MaxPooling2D, GlobalMaxPooling2D
import tensorflow as tf
import tensorflow_datasets as tfds
from tensorflow.keras.initializers import glorot_uniform
from tensorflow.keras import layers
from tensorflow.keras.models import Model
from dragonfly import load_config, multiobjective_maximise_functions,multiobjective_minimise_functions
from dragonfly import maximise_function,minimise_function
import shutil
from numpy.random import seed




def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def train_input_fn(batch_size):
    data = tfds.load('mnist', as_supervised=True)
    train_data = data['train']
    train_data = train_data.map(preprocess).shuffle(500).batch(batch_size)
    return train_data




def identity_block(X, f, filters, stage, block):
    """
    Implementation of the identity block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network

    Returns:
    X -- output of the identity block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = "res" + str(stage) + block + "_branch"
    bn_name_base = "bn" + str(stage) + block + "_branch"

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value. You'll need this later to add back to the main path.
    X_shortcut = X

    # First component of main path
    X = Conv2D(filters=F1, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2a", kernel_initializer=glorot_uniform(seed=0))(X)
    # valid mean no padding / glorot_uniform equal to Xaiver initialization - Steve

    X = BatchNormalization(axis=3, name=bn_name_base + "2a")(X)
    X = Activation("relu")(X)
    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(filters=F2, kernel_size=(f, f), strides=(1, 1), padding="same",
               name=conv_name_base + "2b", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2b")(X)
    X = Activation("relu")(X)
    # Third component of main path (≈2 lines)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = Conv2D(filters=F3, kernel_size=(1, 1), strides=(1, 1), padding="valid",
               name=conv_name_base + "2c", kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + "2c")(X)

    X = Add()([X, X_shortcut])
    X = Activation("relu")(X)
    ### END CODE HERE ###

    return X


def convolutional_block(X, f, filters, stage, block, s=2):
    """
    Implementation of the convolutional block as defined in Figure 4

    Arguments:
    X -- input tensor of shape (m, n_H_prev, n_W_prev, n_C_prev)
    f -- integer, specifying the shape of the middle CONV's window for the main path
    filters -- python list of integers, defining the number of filters in the CONV layers of the main path
    stage -- integer, used to name the layers, depending on their position in the network
    block -- string/character, used to name the layers, depending on their position in the network
    s -- Integer, specifying the stride to be used

    Returns:
    X -- output of the convolutional block, tensor of shape (n_H, n_W, n_C)
    """

    # defining name basis
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    # Retrieve Filters
    F1, F2, F3 = filters

    # Save the input value
    X_shortcut = X

    ##### MAIN PATH #####
    # First component of main path
    X = Conv2D(F1, (1, 1), strides=(s, s), name=conv_name_base + '2a', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2a')(X)
    X = Activation('relu')(X)

    ### START CODE HERE ###

    # Second component of main path (≈3 lines)
    X = Conv2D(F2, (f, f), strides=(1, 1), name=conv_name_base + '2b', padding='same',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2b')(X)
    X = Activation('relu')(X)

    # Third component of main path (≈2 lines)
    X = Conv2D(F3, (1, 1), strides=(1, 1), name=conv_name_base + '2c', padding='valid',
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name=bn_name_base + '2c')(X)

    ##### SHORTCUT PATH #### (≈2 lines)
    X_shortcut = Conv2D(F3, (1, 1), strides=(s, s), name=conv_name_base + '1', padding='valid',
                        kernel_initializer=glorot_uniform(seed=0))(X_shortcut)
    X_shortcut = BatchNormalization(axis=3, name=bn_name_base + '1')(X_shortcut)

    # Final step: Add shortcut value to main path, and pass it through a RELU activation (≈2 lines)
    X = layers.add([X, X_shortcut])
    X = Activation('relu')(X)

    ### END CODE HERE ###

    return X

def ResNet50(input_shape, classes):
    """
    Implementation of the popular ResNet50 the following architecture:
    CONV2D -> BATCHNORM -> RELU -> MAXPOOL -> CONVBLOCK -> IDBLOCK*2 -> CONVBLOCK -> IDBLOCK*3
    -> CONVBLOCK -> IDBLOCK*5 -> CONVBLOCK -> IDBLOCK*2 -> AVGPOOL -> TOPLAYER

    Arguments:
    input_shape -- shape of the images of the dataset
    classes -- integer, number of classes

    Returns:
    model -- a Model() instance in Keras
    """

    # Define the input as a tensor with shape input_shape
    global KS
    global DENSE_UNIT
    global DROP_OUT
    X_input = Input(input_shape)

    # Zero-Padding
    X = ZeroPadding2D((3, 3))(X_input)

    # Stage 1
    X = Conv2D(filters=64, kernel_size=(7, 7), strides=(2, 2), name="conv",
               kernel_initializer=glorot_uniform(seed=0))(X)
    X = BatchNormalization(axis=3, name="bn_conv1")(X)
    X = Activation("relu")(X)
    X = MaxPooling2D(pool_size=(3, 3), strides=(2, 2))(X)

    # Stage 2
    X = convolutional_block(X, f=KS, filters=[64, 64, 256], stage=2, block="a", s=1)
    X = identity_block(X, f=KS, filters=[64, 64, 256], stage=2, block="b")
    X = identity_block(X, f=KS, filters=[64, 64, 256], stage=2, block="c")
    ### START CODE HERE ###

    # Stage 3 (≈4 lines)
    # The convolutional block uses three set of filters of size [128,128,512], "f" is 3, "s" is 2 and the block is "a".
    # The 3 identity blocks use three set of filters of size [128,128,512], "f" is 3 and the blocks are "b", "c" and "d".
    X = convolutional_block(X, f=KS, filters=[128, 128, 512], stage=3, block="a", s=1)
    X = identity_block(X, f=KS, filters=[128, 128, 512], stage=3, block="b")
    X = identity_block(X, f=KS, filters=[128, 128, 512], stage=3, block="c")
    X = identity_block(X, f=KS, filters=[128, 128, 512], stage=3, block="d")

    # Stage 4 (≈6 lines)
    # The convolutional block uses three set of filters of size [256, 256, 1024], "f" is 3, "s" is 2 and the block is "a".
    # The 5 identity blocks use three set of filters of size [256, 256, 1024], "f" is 3 and the blocks are "b", "c", "d", "e" and "f".
    X = convolutional_block(X, f=KS, filters=[256, 256, 1024], stage=4, block="a", s=2)
    X = identity_block(X, f=KS, filters=[256, 256, 1024], stage=4, block="b")
    X = identity_block(X, f=KS, filters=[256, 256, 1024], stage=4, block="c")
    X = identity_block(X, f=KS, filters=[256, 256, 1024], stage=4, block="d")
    X = identity_block(X, f=KS, filters=[256, 256, 1024], stage=4, block="e")
    X = identity_block(X, f=KS, filters=[256, 256, 1024], stage=4, block="f")

    # Stage 5 (≈3 lines)
    # The convolutional block uses three set of filters of size [512, 512, 2048], "f" is 3, "s" is 2 and the block is "a".
    # The 2 identity blocks use three set of filters of size [256, 256, 2048], "f" is 3 and the blocks are "b" and "c".
    X = convolutional_block(X, f=KS, filters=[512, 512, 2048], stage=5, block="a", s=2)
    X = identity_block(X, f=KS, filters=[512, 512, 2048], stage=5, block="b")
    X = identity_block(X, f=KS, filters=[512, 512, 2048], stage=5, block="c")

    # filters should be [256, 256, 2048], but it fail to be graded. Use [512, 512, 2048] to pass the grading

    # AVGPOOL (≈1 line). Use "X = AveragePooling2D(...)(X)"
    # The 2D Average Pooling uses a window of shape (2,2) and its name is "avg_pool".
    X = AveragePooling2D(pool_size=(2, 2), padding="same")(X)

    ### END CODE HERE ###

    # output layer
    X = Flatten()(X)
    X = Dense(units=DENSE_UNIT, activation="relu", kernel_initializer='zeros')(X)
    X = tf.keras.layers.Dropout(DROP_OUT)(X)
    X = Dense(classes, activation="softmax", name="fc" + str(classes), kernel_initializer=glorot_uniform(seed=0))(X)

    # Create model
    model = Model(inputs=X_input, outputs=X, name="ResNet50")

    return model




def model_fn(features, labels, mode):
    global OPTIMIZER
    global LEARNING_RATE
    model = ResNet50(input_shape=(28,28,1),classes=10)
    model = tf.keras.Sequential([
        model
    ])
    logits = model(features, training=False)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    if OPTIMIZER == 'adam':
        optimizer = tf.compat.v1.train.AdamOptimizer(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == 'grad':
        optimizer = tf.compat.v1.train.GradientDescentOptimizer(learning_rate=LEARNING_RATE)
    elif OPTIMIZER == 'rmsp':
        optimizer = tf.compat.v1.train.RMSPropOptimizer(learning_rate=LEARNING_RATE)

    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)(labels, logits)
    loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)


    class reportHook(tf.train.SessionRunHook):
        def __init__(self, loss):
            self.loss = loss

        def before_run(self, run_context):
            return tf.estimator.SessionRunArgs(self.loss)

        def after_run(self, run_context, run_values):
            self.result = run_values.results
            global final_loss
            final_loss = self.result

        def end(self,session):
            global final_loss
            final_loss = self.result


    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(
            loss, tf.compat.v1.train.get_or_create_global_step()),
        training_hooks=[reportHook(loss)])


def too_long_and_stop():
    curr = time.time()
    global global_start_time
    diff = curr - global_start_time
    if diff > 1200:
        return True
    else:
        return False


def loss_eval(x):
    global BATCH_SIZE
    BATCH_SIZE = x[0]
    global LEARNING_RATE
    LEARNING_RATE = x[1]
    global DROP_OUT
    DROP_OUT = x[2]
    global DENSE_UNIT
    DENSE_UNIT = x[3]
    global OPTIMIZER
    OPTIMIZER = x[4]
    global KS
    KS = x[5]

    my_config = tf.ConfigProto( 
        inter_op_parallelism_threads=x[6],
        intra_op_parallelism_threads=x[7],
        graph_options=tf.GraphOptions(
            build_cost_model=x[8],
            infer_shapes=x[9],
            place_pruned_graph=x[10],
            enable_bfloat16_sendrecv=x[11],
            optimizer_options=tf.OptimizerOptions(
                do_common_subexpression_elimination=x[12],
                max_folded_constant_in_bytes=x[13],
                do_function_inlining=x[14],
                global_jit_level=x[15]
                )))
    config = tf.estimator.RunConfig(save_summary_steps=1,
                                save_checkpoints_steps=FLAGS.save_ckpt_steps,
                                save_checkpoints_secs=None,
                                log_step_count_steps=1,
                                session_config=my_config)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=model_dir, config=config)
    stop_hook = tf.estimator.experimental.make_early_stopping_hook(classifier, should_stop_fn=too_long_and_stop,run_every_secs=60)
    global global_start_time
    global_start_time = time.time()
    tf.estimator.train_and_evaluate(classifier,
        train_spec=tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(BATCH_SIZE), max_steps=FLAGS.train_steps,hooks=[stop_hook]),
        eval_spec=tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(BATCH_SIZE), steps=10))
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print("clean is finish")

    global final_loss
    return -float(final_loss)


def runtime_eval(x):
    global BATCH_SIZE
    BATCH_SIZE = x[0]
    global LEARNING_RATE
    LEARNING_RATE = x[1]
    global DROP_OUT
    DROP_OUT = x[2]
    global DENSE_UNIT
    DENSE_UNIT = x[3]
    global OPTIMIZER
    OPTIMIZER = x[4]
    global KS
    KS = x[5]

    my_config = tf.ConfigProto( 
        inter_op_parallelism_threads=x[6],
        intra_op_parallelism_threads=x[7],
        graph_options=tf.GraphOptions(
            build_cost_model=x[8],
            infer_shapes=x[9],
            place_pruned_graph=x[10],
            enable_bfloat16_sendrecv=x[11],
            optimizer_options=tf.OptimizerOptions(
                do_common_subexpression_elimination=x[12],
                max_folded_constant_in_bytes=x[13],
                do_function_inlining=x[14],
                global_jit_level=x[15]
                )))
    config = tf.estimator.RunConfig(save_summary_steps=1,
                                save_checkpoints_steps=FLAGS.save_ckpt_steps,
                                save_checkpoints_secs=None,
                                log_step_count_steps=1,
                                session_config=my_config)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=model_dir, config=config)
    stop_hook = tf.estimator.experimental.make_early_stopping_hook(classifier, should_stop_fn=too_long_and_stop,run_every_secs=60)
    start_time = time.time()
    global global_start_time
    global_start_time = start_time
    tf.estimator.train_and_evaluate(classifier,
        train_spec=tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(BATCH_SIZE), max_steps=FLAGS.train_steps,hooks=[stop_hook]),
        eval_spec=tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(BATCH_SIZE), steps=10))
    current_time = time.time()
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print("clean is finish")

    return -float(current_time - start_time)






FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_integer('save_ckpt_steps', 160, 'save ckpt per n steps')
tf.app.flags.DEFINE_string('model_dir', './estimator', 'model_dir')
tf.app.flags.DEFINE_integer('train_steps', 150, 'train_steps')
tf.app.flags.DEFINE_string('dataset', 'mnist', 'specify dataset')

model_dir = FLAGS.model_dir

seed(0)
tf.compat.v1.random.set_random_seed(0)
tf.compat.v1.disable_eager_execution()
tfds.disable_progress_bar()
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)

#global variables:
BUFFER_SIZE = 10000
IMG_SIZE = 28


BATCH_SIZE = 32
LEARNING_RATE = 1e-3
DROP_OUT = 0.1
DENSE_UNIT = 32
OPTIMIZER = 'adam'
KS = 3
final_loss = 0.0
global_start_time = time.time()

batch_list = [8,16,32,64,128]
LR_list = [7.5e-1,5e-1,2.5e-1,1e-1,7.5e-2,5e-2]
DROP_list = [1e-1,2e-1,3e-1,4e-1,5e-1]
DENSE_list = [32,64,128,256,512]
OPTIMIZER_list = ['adam','grad','rmsp']
KS_list = [2,3,4,5]
inter_list = [1,2,3,4]
intra_list = [2,4,6,8,10,12]
build_cost_model_list = [0,2,4,6,8]
infer_shapes_list = [0,1]
place_pruned_graph_list = [0,1]
enable_bfloat16_sendrecv_list = [0,1]
do_common_subexpression_elimination_list = [0,1]
max_folded_constant_list = [2,4,6,8,10]
do_function_inlining_list = [0,1]
global_jit_level_list = [0,1,2]





domain_vars = [{'type': 'discrete_numeric', 'items': batch_list},
                {'type': 'discrete_numeric', 'items': LR_list},
                {'type': 'discrete_numeric', 'items': DROP_list},
                {'type': 'discrete_numeric', 'items': DENSE_list},
                {'type': 'discrete', 'items': OPTIMIZER_list},
                {'type': 'discrete_numeric', 'items': KS_list},
                {'type': 'discrete_numeric', 'items': inter_list},
                {'type': 'discrete_numeric', 'items': intra_list},
                {'type': 'discrete_numeric', 'items': build_cost_model_list},
                {'type': 'discrete_numeric', 'items': infer_shapes_list},
                {'type': 'discrete_numeric', 'items': place_pruned_graph_list},
                {'type': 'discrete_numeric', 'items': enable_bfloat16_sendrecv_list},
                {'type': 'discrete_numeric', 'items': do_common_subexpression_elimination_list},
                {'type': 'discrete_numeric', 'items': max_folded_constant_list},
                {'type': 'discrete_numeric', 'items': do_function_inlining_list},
                {'type': 'discrete_numeric', 'items': global_jit_level_list}
                ]
config_params = {'domain': domain_vars}
config = load_config(config_params)
max_num_evals = 60 * 60 * 12
moo_objectives = [runtime_eval, loss_eval]
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type='realtime',config=config)
f = open("./output.log","w+")
print(pareto_opt_pts,file=f)
print("\n",file=f)
print(pareto_opt_vals,file=f)
print("\n",file=f)
print(history,file=f)




