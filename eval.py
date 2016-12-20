#! /usr/bin/env python3

import tensorflow as tf
import numpy as np
import logging
import sys
import os
import time
import datetime
import data_helpers
from text_cnn import TextCNN
from tensorflow.contrib import learn
import csv


# Use logging facilities to make output more friendly
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# Parameters
# ==================================================

# Data Parameters
tf.flags.DEFINE_string("positive_data_file", "./data/kinopoisk/polarity.pos", "Data source for the positive data.")
tf.flags.DEFINE_string("negative_data_file", "./data/kinopoisk/polarity.neg", "Data source for the positive data.")

# Eval Parameters
tf.flags.DEFINE_integer("batch_size", 64, "Batch Size (default: 64)")
tf.flags.DEFINE_string("checkpoint_dir", "", "Checkpoint directory from training run")
tf.flags.DEFINE_boolean("eval_train", False, "Evaluate on all training data")
tf.flags.DEFINE_string("eval_input", None, "Evaluate the provided input")

# Output Parameters
tf.flags.DEFINE_boolean("out_stdout", False, "Output to stdout")
tf.flags.DEFINE_string("out_path", None, "Output path for prediction results")

# Misc Parameters
tf.flags.DEFINE_boolean("allow_soft_placement", True, "Allow device soft device placement")
tf.flags.DEFINE_boolean("log_device_placement", False, "Log placement of ops on devices")


FLAGS = tf.flags.FLAGS
FLAGS._parse_flags()
logger.info("\nParameters:")
for attr, value in sorted(FLAGS.__flags.items()):
    logger.info("{}={}".format(attr.upper(), value))
logger.info("")

# CHANGE THIS: Load data. Load your own data here
if FLAGS.eval_train:
    x_raw, y_test = data_helpers.load_data_and_labels(FLAGS.positive_data_file, FLAGS.negative_data_file)
    y_test = np.argmax(y_test, axis=1)
elif FLAGS.eval_input:
    x_raw = [FLAGS.eval_input]
    y_test = [1]
else:
    x_raw = ["Шикарно! Обожаю Киану. Первая часть была топчиком, оживила жанр как боевик. Чисто кино для отдыха.",
             "надеюсь на этот фильм, а то так жанр боевика в наши дни умер по сути…",
             "Постер суперский! Я ну очень надеюсь, что не перегрузят кино персонажами и 2-я глава будет не хуже первой",
             "Очень большие надежды, поход в кино уже запланирован.",
             "Я был тронут, поражен, мурашки буквально сожрали мое несчастное тело, которое и так в это время боролось с другими реакциями моего организма.",
             "Подростки какие-то тоже глупые и унылые, как-то особо привлекательности в них нет",
             "Насчет продажности и т. п. тоже можно было бы циничнее все оформить.",
             "Однако время шло, деньги собирались, съёмки, вроде бы, велись, но по срокам никто-ничего не говорил.",
             "Не нужно поддерживать таких режиссёров.",
             "Общее впечатление: опять ругал себя за то что потратил 90 минут времени.",
             ]
    y_test = [1, 1, 1, 1, 1, 0, 0, 0, 0, 0]

# Map data into vocabulary
vocab_path = os.path.join(FLAGS.checkpoint_dir, "..", "vocab")
vocab_processor = learn.preprocessing.VocabularyProcessor.restore(vocab_path)
x_test = np.array(list(vocab_processor.transform(x_raw)))

logger.info("\nEvaluating...\n")

# Evaluation
# ==================================================
checkpoint_file = tf.train.latest_checkpoint(FLAGS.checkpoint_dir)
graph = tf.Graph()
with graph.as_default():
    session_conf = tf.ConfigProto(
      allow_soft_placement=FLAGS.allow_soft_placement,
      log_device_placement=FLAGS.log_device_placement)
    sess = tf.Session(config=session_conf)
    with sess.as_default():
        # Load the saved meta graph and restore variables
        saver = tf.train.import_meta_graph("{}.meta".format(checkpoint_file))
        saver.restore(sess, checkpoint_file)

        # Get the placeholders from the graph by name
        input_x = graph.get_operation_by_name("input_x").outputs[0]
        # input_y = graph.get_operation_by_name("input_y").outputs[0]
        dropout_keep_prob = graph.get_operation_by_name("dropout_keep_prob").outputs[0]

        # Tensors we want to evaluate
        predictions = graph.get_operation_by_name("output/predictions").outputs[0]

        # Generate batches for one epoch
        batches = data_helpers.batch_iter(list(x_test), FLAGS.batch_size, 1, shuffle=False)

        # Collect the predictions here
        all_predictions = []

        for x_test_batch in batches:
            batch_predictions = sess.run(predictions, {input_x: x_test_batch, dropout_keep_prob: 1.0})
            all_predictions = np.concatenate([all_predictions, batch_predictions])

# Print accuracy if y_test is defined
if y_test is not None:
    correct_predictions = float(sum(all_predictions == y_test))
    logger.info("Total number of test examples: {}".format(len(y_test)))
    logger.info("Accuracy: {:g}".format(correct_predictions/float(len(y_test))))

# Save the evaluation to a csv
predictions_human_readable = np.column_stack((np.array(x_raw), all_predictions))
if FLAGS.out_stdout:
    csv.writer(sys.stdout).writerows(predictions_human_readable)
else:
    if FLAGS.out_path:
        out_path = FLAGS.out_path
    else:
        out_path = os.path.join(FLAGS.checkpoint_dir, "..", "prediction.csv")
    logger.info("Saving evaluation to {0}".format(out_path))
    with open(out_path, 'w') as f:
        csv.writer(f).writerows(predictions_human_readable)
