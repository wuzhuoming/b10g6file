from __future__ import absolute_import, division, print_function, unicode_literals
import time
import json
import os
from dragonfly import load_config, multiobjective_maximise_functions,multiobjective_minimise_functions
from dragonfly import maximise_function,minimise_function
import tensorflow as tf
import tensorflow_datasets as tfds
from numpy.random import seed
from tensorflow.keras.initializers import glorot_uniform
import shutil


def preprocess(image, label):
    image = tf.cast(image, tf.float32)
    image = tf.image.grayscale_to_rgb(image)
    image = (image / 127.5) - 1
    image = tf.image.resize(image, (IMG_SIZE, IMG_SIZE))
    return image, label


def train_input_fn(batch_size, dataset):
    data = tfds.load(dataset, as_supervised=True)
    train_data = data['train']
    train_data = train_data.map(preprocess).batch(batch_size)
    return train_data


def get_default_params():
    return {
        "BATCH_SIZE":32,
        "LEARNING_RATE":1e-3,
        "inter_op_parallelism_threads":1,
        "intra_op_parallelism_threads":2
    }


def model_fn(features, labels, mode):
    model = tf.keras.Sequential([
        tf.keras.layers.Conv2D(input_shape=(IMG_SIZE, IMG_SIZE, 3), filters=64, kernel_size=(3, 3), padding="same",
                               activation="relu", kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=64, kernel_size=(3, 3), padding="same", activation="relu",
                               kernel_initializer='zeros'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu",
                               kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=128, kernel_size=(3, 3), padding="same", activation="relu",
                               kernel_initializer='zeros'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu",
                               kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu",
                               kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=256, kernel_size=(3, 3), padding="same", activation="relu",
                               kernel_initializer='zeros'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu",
                               kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu",
                               kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu",
                               kernel_initializer='zeros'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu",
                               kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu",
                               kernel_initializer='zeros'),
        tf.keras.layers.Conv2D(filters=512, kernel_size=(3, 3), padding="same", activation="relu",
                               kernel_initializer='zeros'),
        tf.keras.layers.MaxPooling2D(pool_size=(2, 2), strides=(2, 2)),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(units=256, activation="relu", kernel_initializer='zeros'),
        tf.keras.layers.Dense(units=256, activation="relu", kernel_initializer='zeros'),
        tf.keras.layers.Dense(units=12, activation="softmax")
    ])

    logits = model(features, training=True)

    if mode == tf.estimator.ModeKeys.PREDICT:
        predictions = {'logits': logits}
        return tf.estimator.EstimatorSpec(labels=labels, predictions=predictions)

    optimizer = tf.compat.v1.train.GradientDescentOptimizer(
        learning_rate=LEARNING_RATE)
    loss = tf.keras.losses.SparseCategoricalCrossentropy(
        from_logits=True, reduction=tf.losses.Reduction.NONE)(labels, logits)
    loss = tf.reduce_sum(loss) * (1. / BATCH_SIZE)
    if mode == tf.estimator.ModeKeys.EVAL:
        return tf.estimator.EstimatorSpec(mode, loss=loss)

    class NNIReportHook(tf.train.SessionRunHook):
        def __init__(self, loss):
            self.loss = loss

        def before_run(self, run_context):
            return tf.estimator.SessionRunArgs(self.loss)

        def after_run(self, run_context, run_values):
            self.result = run_values.results

        def end(self,session):
            global final_loss
            final_loss = self.result
            

    return tf.estimator.EstimatorSpec(
        mode=mode,
        loss=loss,
        train_op=optimizer.minimize(
            loss, tf.compat.v1.train.get_or_create_global_step()),
        training_hooks=[NNIReportHook(loss)])


def loss_eval(x):
    global BATCH_SIZE
    BATCH_SIZE = x[0]
    global LEARNING_RATE
    LEARNING_RATE = x[1]
    inter_op_parallelism_threads = x[2]
    intra_op_parallelism_threads = x[3]
    my_config = tf.ConfigProto( 
        inter_op_parallelism_threads=inter_op_parallelism_threads,
        intra_op_parallelism_threads=intra_op_parallelism_threads)
    config = tf.estimator.RunConfig(save_summary_steps=1,
                                save_checkpoints_steps=FLAGS.save_ckpt_steps,
                                save_checkpoints_secs=None,
                                log_step_count_steps=1,
                                session_config=my_config)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=model_dir, config=config)
    tf.estimator.train_and_evaluate(classifier,
        train_spec=tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(BATCH_SIZE, FLAGS.dataset), max_steps=FLAGS.train_steps),
        eval_spec=tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(BATCH_SIZE, FLAGS.dataset), steps=10))
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print("clean is finish")

    global current_eval
    current_eval = current_eval + 1
    print("current_Eval : %d" % (current_eval))

    global final_loss
    return -float(final_loss)

def runtime_eval(x):
    global BATCH_SIZE
    BATCH_SIZE = x[0]
    global LEARNING_RATE
    LEARNING_RATE = x[1]
    inter_op_parallelism_threads = x[2]
    intra_op_parallelism_threads = x[3]
    my_config = tf.ConfigProto( 
        inter_op_parallelism_threads=inter_op_parallelism_threads,
        intra_op_parallelism_threads=intra_op_parallelism_threads)
    config = tf.estimator.RunConfig(save_summary_steps=1,
                                save_checkpoints_steps=FLAGS.save_ckpt_steps,
                                save_checkpoints_secs=None,
                                log_step_count_steps=1,
                                session_config=my_config)
    classifier = tf.estimator.Estimator(
        model_fn=model_fn, model_dir=model_dir, config=config)
    start_time = time.time()
    tf.estimator.train_and_evaluate(classifier,
        train_spec=tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(BATCH_SIZE, FLAGS.dataset), max_steps=FLAGS.train_steps),
        eval_spec=tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(BATCH_SIZE, FLAGS.dataset), steps=10))
    current_time = time.time()
    if os.path.exists(model_dir):
        shutil.rmtree(model_dir)
        print("clean is finish")
    return -float(current_time - start_time)



seed(0)
tf.compat.v1.random.set_random_seed(0)
tf.compat.v1.disable_eager_execution()
tfds.disable_progress_bar()
BUFFER_SIZE = 10000

params = get_default_params()
IMG_SIZE = 48  # All images will be resized to IMG_SIZE*IMG_SIZE
# 160 for cats_vs_dogs(input_shape:None, None, 3); 28 for mnist(input_shape:28, 28, 1)
IMG_CLASS = 10

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('worker', 'localhost:3001,localhost:3002', 'specify workers in the cluster')
tf.app.flags.DEFINE_string('dataset', 'mnist', 'specify dataset')
tf.app.flags.DEFINE_integer('task_index', 0, 'task_index')
tf.app.flags.DEFINE_string('model_dir', './estimator-original', 'model_dir')
tf.app.flags.DEFINE_integer('save_ckpt_steps', 150, 'save ckpt per n steps')
tf.app.flags.DEFINE_boolean('use_original_ckpt', True, 'use original ckpt')
tf.app.flags.DEFINE_integer('train_steps', 100, 'train_steps')

model_dir = FLAGS.model_dir
# To solve path problem when model_dir of current worker and worker 0 are different


BATCH_SIZE = 32
LEARNING_RATE = 1e-3
final_loss = 0.0
current_eval = 0


tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.INFO)
# strategy = None
# inter_op_parallelism_threads = params['inter_op_parallelism_threads']
# intra_op_parallelism_threads = params['intra_op_parallelism_threads']
# my_config = tf.ConfigProto( 
#     inter_op_parallelism_threads=inter_op_parallelism_threads,
#     intra_op_parallelism_threads=intra_op_parallelism_threads)

# config = tf.estimator.RunConfig(save_summary_steps=1,
#                                 save_checkpoints_steps=FLAGS.save_ckpt_steps,
#                                 save_checkpoints_secs=None,
#                                 log_step_count_steps=1,
#                                 session_config=my_config)

# classifier = tf.estimator.Estimator(
#     model_fn=model_fn, model_dir=model_dir, config=config)

# tf.estimator.train_and_evaluate(
#     classifier,
#     train_spec=tf.estimator.TrainSpec(input_fn=lambda: train_input_fn(BATCH_SIZE, FLAGS.dataset), max_steps=FLAGS.train_steps),
#     eval_spec=tf.estimator.EvalSpec(input_fn=lambda: train_input_fn(BATCH_SIZE, FLAGS.dataset), steps=10)
# )

batch_list = [8,16,24,40,56,80]
LR_list = [5e-2,1e-1,1.5e-1,2.5e-1,4e-1,6e-1]
intra_list = [2,4,6,8,10,12]
domain_vars = [{'type': 'discrete_numeric', 'items': batch_list},
                {'type': 'discrete_numeric', 'items': LR_list},
                {'type': 'int', 'min': 1, 'max': 6},
                {'type': 'discrete_numeric', 'items': intra_list}]
config_params = {'domain': domain_vars}
config = load_config(config_params)
max_num_evals = 60 * 60 * 4
moo_objectives = [runtime_eval, loss_eval]
pareto_opt_vals, pareto_opt_pts, history = multiobjective_maximise_functions(moo_objectives, config.domain,max_num_evals,capital_type='realtime',config=config)
f = open("./output.log","w+")
print(pareto_opt_pts,file=f)
print("\n",file=f)
print(pareto_opt_vals,file=f)
print("\n",file=f)
print(history,file=f)
print("\n",file=f)
print(current_eval,file=f)

# Delete the checkpoint and summary for next trial
