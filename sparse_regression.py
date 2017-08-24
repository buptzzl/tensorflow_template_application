#!/usr/bin/env python
# -*- encoding: utf-8 -*-

import datetime
import json
import logging
import math
import numpy as np
import os
import pprint
from sklearn import metrics
import tensorflow as tf
from tensorflow.contrib.session_bundle import exporter
from tensorflow.python.saved_model import builder as saved_model_builder
from tensorflow.python.saved_model import signature_constants
from tensorflow.python.saved_model import signature_def_utils
from tensorflow.python.saved_model import tag_constants
from tensorflow.python.saved_model import utils
from tensorflow.python.util import compat
from tensorflow.python.ops import math_ops

START_TIME = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
# Define hyperparameters
flags = tf.app.flags
FLAGS = flags.FLAGS
flags.DEFINE_boolean("enable_colored_log", False, "Enable colored log")
flags.DEFINE_string("train_tfrecords_file",
                    "./data/a8a/a8a_train.libsvm.tfrecords",
                    "The glob pattern of train TFRecords files")
flags.DEFINE_string("validate_tfrecords_file",
                    "./data/a8a/a8a_test.libsvm.tfrecords",
                    "The glob pattern of validate TFRecords files")
flags.DEFINE_integer("feature_size", 124, "Number of feature size")
flags.DEFINE_integer("label_size", 2, "Number of label size")
flags.DEFINE_float("learning_rate", 0.01, "The learning rate")
flags.DEFINE_integer("epoch_number", 10, "Number of epochs to train")
flags.DEFINE_integer("batch_size", 1024, "The batch size of training")
flags.DEFINE_integer("validate_batch_size", 1024,
                     "The batch size of validation")
flags.DEFINE_integer("batch_thread_number", 1,
                     "Number of threads to read data")
flags.DEFINE_integer("min_after_dequeue", 100,
                     "The minimal number after dequeue")
flags.DEFINE_string("checkpoint_path", "./checkpoint/" + START_TIME,
                    "The path of checkpoint")
#flags.DEFINE_string("checkpoint_clean", True,  "whether delete the path of checkpoint")
flags.DEFINE_string("output_path", "output/" + START_TIME,
                    "The path of tensorboard event files")
flags.DEFINE_string("model", "dnn", "Support dnn, lr, wide_and_deep")
flags.DEFINE_string("model_network", "128 32 8", "The neural network of model")
flags.DEFINE_boolean("enable_bn", False, "Enable batch normalization or not")
flags.DEFINE_float("bn_epsilon", 0.001, "The epsilon of batch normalization")
flags.DEFINE_boolean("enable_dropout", False, "Enable dropout or not")
flags.DEFINE_float("dropout_keep_prob", 0.5, "The dropout keep prob")
flags.DEFINE_boolean("enable_lr_decay", False, "Enable learning rate decay")
flags.DEFINE_float("lr_decay_rate", 0.96, "Learning rate decay rate")
flags.DEFINE_string("optimizer", "adagrad", "The optimizer to train")
flags.DEFINE_integer("steps_to_validate", 10,
                     "Steps to validate and print state")
flags.DEFINE_string("mode", "train", "Support train, export, inference")
flags.DEFINE_string("saved_model_path", "./model_save/" + START_TIME,
                    "The path of the saved model")
flags.DEFINE_string("model_path", "./model/" + START_TIME, "The path of the model")
flags.DEFINE_integer("model_version", 1, "The version of the model")
flags.DEFINE_string("inference_test_file", "./data/a8a_test.libsvm",
                    "The test file for inference")
flags.DEFINE_string("inference_result_file", "./inference_result.txt",
                    "The result file from inference")
flags.DEFINE_boolean("benchmark_mode", False,
                     "Reduce extra computation in benchmark mode")


def main():
  # Get hyperparameters
  if FLAGS.enable_colored_log:
    import coloredlogs
    coloredlogs.install()
  logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(lineno)d %(message)s")
  FEATURE_SIZE = FLAGS.feature_size
  LABEL_SIZE = FLAGS.label_size
  EPOCH_NUMBER = FLAGS.epoch_number
  if EPOCH_NUMBER <= 0:
    EPOCH_NUMBER = None
  BATCH_THREAD_NUMBER = FLAGS.batch_thread_number
  MIN_AFTER_DEQUEUE = FLAGS.min_after_dequeue
  BATCH_CAPACITY = BATCH_THREAD_NUMBER * FLAGS.batch_size + MIN_AFTER_DEQUEUE
  MODE = FLAGS.mode
  MODEL = FLAGS.model
  OPTIMIZER = FLAGS.optimizer
  CHECKPOINT_PATH = FLAGS.checkpoint_path
  if not CHECKPOINT_PATH.startswith("fds://") and not os.path.exists(
      CHECKPOINT_PATH):
    os.makedirs(CHECKPOINT_PATH)
  CHECKPOINT_FILE = CHECKPOINT_PATH + "/checkpoint.ckpt"
  LATEST_CHECKPOINT = tf.train.latest_checkpoint(CHECKPOINT_PATH)
  OUTPUT_PATH = FLAGS.output_path
  if not OUTPUT_PATH.startswith("fds://") and not os.path.exists(OUTPUT_PATH):
    os.makedirs(OUTPUT_PATH)
  pprint.PrettyPrinter().pprint(FLAGS.__flags)

  # dubug: 统计处理的样本数
  sample_size = tf.Variable(0, name="sample_size", trainable=False)
  sample_inc_op = sample_size.assign_add(FLAGS.batch_size)  #  统计训练处理的样本数，对应图的哪一步?
  validate_size = tf.Variable(0, name="validate_size", trainable=False)
  validate_inc_op = validate_size.assign_add(FLAGS.batch_size)  

  # Read TFRecords files for training
  def read_and_decode(filename_queue):
    reader = tf.TFRecordReader()
    _, serialized_example = reader.read(filename_queue)
    # debug:
    #logging.info("%s read %s instances" % (filename_queue, reader.num_records_produced()))
    return serialized_example

  # Read TFRecords files for training
  filename_queue = tf.train.string_input_producer(
      tf.train.match_filenames_once(FLAGS.train_tfrecords_file),
      num_epochs=EPOCH_NUMBER)
  serialized_example = read_and_decode(filename_queue)

  batch_serialized_example = tf.train.shuffle_batch(
      [serialized_example],
      batch_size=FLAGS.batch_size,
      num_threads=BATCH_THREAD_NUMBER,
      capacity=BATCH_CAPACITY,
      min_after_dequeue=MIN_AFTER_DEQUEUE)
  features = tf.parse_example(
      batch_serialized_example,
      features={
          "label": tf.FixedLenFeature([], tf.float32),
          "ids": tf.VarLenFeature(tf.int64),
          "values": tf.VarLenFeature(tf.float32),
      })
  batch_labels = features["label"]
  batch_ids = features["ids"]
  batch_values = features["values"]
  ## debug: can't print
  #batch_serialized_example = tf.Print(batch_serialized_example, [batch_labels],
  #                                    "debug read label", summarize=10000, first_n=10000)

  ## dubug: get tfiles samples:
  #for fn in filename_queue:
  #  logging.info("sample file: {}, samples: {}".format(fn,
  #               sum(1 for _ in tf.python_io.tf_record_iterator(fn))))

  # Read TFRecords file for validation 
  ## 异常：不能支持循环读入验证数据集导致抛 OutOfRangeError
  validate_filename_queue = tf.train.string_input_producer(
      tf.train.match_filenames_once(FLAGS.validate_tfrecords_file),
      num_epochs=EPOCH_NUMBER)
  validate_serialized_example = read_and_decode(validate_filename_queue)
  validate_batch_serialized_example = tf.train.shuffle_batch(   # 此处只读了100个样本就终止？
      [validate_serialized_example],
      batch_size=FLAGS.validate_batch_size,
      num_threads=BATCH_THREAD_NUMBER,
      capacity=BATCH_CAPACITY,
      min_after_dequeue=MIN_AFTER_DEQUEUE)
  validate_features = tf.parse_example(
      validate_batch_serialized_example,
      features={
          "label": tf.FixedLenFeature([], tf.float32),
          "ids": tf.VarLenFeature(tf.int64),
          "values": tf.VarLenFeature(tf.float32),
      })
  validate_batch_labels = validate_features["label"]
  validate_batch_ids = validate_features["ids"]
  validate_batch_values = validate_features["values"]

  # Define the model
  input_units = FEATURE_SIZE
  output_units = LABEL_SIZE
  model_network_hidden_units = [int(i) for i in FLAGS.model_network.split()]

  def full_connect(inputs, weights_shape, biases_shape, is_train=True):
    with tf.device("/cpu:0"):
      weights = tf.get_variable(
          "weights", weights_shape, initializer=tf.random_normal_initializer(mean=0.0, stddev=0.01)) # math_ops.abs---))
      biases = tf.get_variable(
          "biases", biases_shape, initializer=tf.random_normal_initializer())
      layer = tf.matmul(inputs, weights) + biases

      if FLAGS.enable_bn and is_train:
        mean, var = tf.nn.moments(layer, axes=[0])
        scale = tf.get_variable(
            "scale", biases_shape, initializer=tf.random_normal_initializer())
        shift = tf.get_variable(
            "shift", biases_shape, initializer=tf.random_normal_initializer())
        layer = tf.nn.batch_normalization(layer, mean, var, shift, scale,
                                          FLAGS.bn_epsilon)
    return layer

  def sparse_full_connect(sparse_ids,
                          sparse_values,
                          weights_shape,
                          biases_shape,
                          is_train=True):
    weights = tf.get_variable(
        "weights", weights_shape, initializer=tf.random_normal_initializer())
    biases = tf.get_variable(
        "biases", biases_shape, initializer=tf.random_normal_initializer())
    return tf.nn.embedding_lookup_sparse(
        weights, sparse_ids, sparse_values, combiner="sum") + biases

  def full_connect_relu(inputs, weights_shape, biases_shape, is_train=True):
    return tf.nn.relu(
        full_connect(inputs, weights_shape, biases_shape, is_train))

  def customized_inference(sparse_ids, sparse_values, is_train=True):
    hidden1_units = 128
    hidden2_units = 32
    hidden3_units = 8

    with tf.variable_scope("input"):
      sparse_layer = sparse_full_connect(sparse_ids, sparse_values,
                                         [input_units, hidden1_units],
                                         [hidden1_units], is_train)
      layer = tf.nn.relu(sparse_layer)
    with tf.variable_scope("layer0"):
      layer = full_connect_relu(layer, [hidden1_units, hidden2_units],
                                [hidden2_units], is_train)
    with tf.variable_scope("layer1"):
      layer = full_connect_relu(layer, [hidden2_units, hidden3_units],
                                [hidden3_units], is_train)
    if FLAGS.enable_dropout and is_train:
      layer = tf.nn.dropout(layer, FLAGS.dropout_keep_prob)
    with tf.variable_scope("output"):
      layer = full_connect(layer, [hidden3_units, output_units],
                           [output_units], is_train)
    return layer

  def dnn_inference(sparse_ids, sparse_values, is_train=True):
    with tf.variable_scope("input"):
      sparse_layer = sparse_full_connect(sparse_ids, sparse_values, [
          input_units, model_network_hidden_units[0]
      ], [model_network_hidden_units[0]], is_train)
      layer = tf.nn.relu(sparse_layer)

    for i in range(len(model_network_hidden_units) - 1):
      with tf.variable_scope("layer{}".format(i)):
        layer = full_connect_relu(layer, [
            model_network_hidden_units[i], model_network_hidden_units[i + 1]
        ], [model_network_hidden_units[i + 1]], is_train)

    with tf.variable_scope("output"):
      layer = full_connect(layer,
                           [model_network_hidden_units[-1],
                            output_units], [output_units], is_train)
    ## debug: 每次加载模型验证显式调用loss 时才触发计算，train_op优化参数时不触发
    #with tf.control_dependencies([sample_inc_op]): # 两种后置计算依赖方式
      #layer = tf.identity(layer, name='layer')
      #layer = tf.Print(layer, [sample_size], "debug sample_size:", summarize=10000, first_n=10000)

    return layer

  def lr_inference(sparse_ids, sparse_values, is_train=True):
    with tf.variable_scope("logistic_regression"):
      layer = sparse_full_connect(sparse_ids, sparse_values,
                                  [input_units, output_units], [output_units])
    return layer

  def wide_and_deep_inference(sparse_ids, sparse_values, is_train=True):
    return lr_inference(sparse_ids, sparse_values, is_train) + dnn_inference(
        sparse_ids, sparse_values, is_train)

  def inference(sparse_ids, sparse_values, is_train=True):
    if MODEL == "dnn":
      return dnn_inference(sparse_ids, sparse_values, is_train)
    elif MODEL == "lr":
      return lr_inference(sparse_ids, sparse_values, is_train)
    elif MODEL == "wide_and_deep":
      return wide_and_deep_inference(sparse_ids, sparse_values, is_train)
    elif MODEL == "customized":
      return customized_inference(sparse_ids, sparse_values, is_train)
    else:
      logging.error("Unknown model: " + MODEL + ", exit now")
      exit(1)

  logging.info("Use the model: {}, model network: {}".format(
      MODEL, FLAGS.model_network))
  logits = inference(batch_ids, batch_values, True)
  # regression: change to regression, ref: TF's tutorial of regression.
  #batch_labels = tf.to_int64(batch_labels)
  #cross_entropy = tf.nn.sparse_softmax_cross_entropy_with_logits(
  #    logits=logits, labels=batch_labels)
  batch_labels = tf.reshape(batch_labels, [FLAGS.batch_size, FLAGS.label_size]) # 解决label与网络输出logits的维度不匹配异常
  #logging.info("batch_labels's shape:", batch_labels.shape, "logits shape:", logits.shape) #, str(batch_labels), str(logits))
  cross_entropy = tf.losses.absolute_difference(labels=batch_labels, predictions=logits)
  loss = tf.reduce_mean(cross_entropy, name="loss")
  global_step = tf.Variable(0, name="global_step", trainable=False)
  if FLAGS.enable_lr_decay:
    logging.info(
        "Enable learning rate decay rate: {}".format(FLAGS.lr_decay_rate))
    starter_learning_rate = FLAGS.learning_rate
    learning_rate = tf.train.exponential_decay(
        starter_learning_rate,
        global_step,
        100000,
        FLAGS.lr_decay_rate,
        staircase=True)
  else:
    learning_rate = FLAGS.learning_rate
  optimizer = get_optimizer(FLAGS.optimizer, learning_rate)
  train_op = optimizer.minimize(loss, global_step=global_step)
  tf.get_variable_scope().reuse_variables()

  # Define accuracy op for train data
  train_accuracy_logits = inference(batch_ids, batch_values, False)
  #train_softmax = tf.nn.softmax(train_accuracy_logits)  # regression
  #train_correct_prediction = tf.equal(
  #    tf.argmax(train_softmax, 1), batch_labels)
  #train_accuracy = tf.reduce_mean(
  #    tf.cast(train_correct_prediction, tf.float32))
  train_accuracy = tf.reduce_mean(math_ops.abs(math_ops.subtract(train_accuracy_logits,
                                                                 batch_labels)))
  ## debug hug loss problem: cannot print out info.
  #logging.info("step: {}, logits: {}, label: {}, accuracy: {}".format(global_step, train_accuracy_logits, batch_labels, train_accuracy))
  #tf.Print(global_step, [train_accuracy_logits], "train_accuracy_logits:") #, 100)
  #tf.Print(batch_labels, [batch_labels], "batch_labels:") # , 100)
  #tf.Print(train_accuracy, [train_accuracy], "train_accuracy:") #, 100)

  ### Define auc op for train data
  #batch_labels = tf.cast(batch_labels, tf.int32)
  #sparse_labels = tf.reshape(batch_labels, [-1, 1])
  #derived_size = tf.shape(batch_labels)[0]
  #indices = tf.reshape(tf.range(0, derived_size, 1), [-1, 1])
  #concated = tf.concat(axis=1, values=[indices, sparse_labels])
  #outshape = tf.stack([derived_size, LABEL_SIZE])
  #new_train_batch_labels = tf.sparse_to_dense(concated, outshape, 1.0, 0.0)
  #_, train_auc = tf.contrib.metrics.streaming_auc(train_softmax,
  #                                                new_train_batch_labels)

  # Define accuracy op for validate data
  validate_accuracy_logits = inference(validate_batch_ids,
                                       validate_batch_values, False)
  validate_accuracy = tf.reduce_mean(math_ops.abs(math_ops.subtract(validate_accuracy_logits,
                                                      validate_batch_labels)))

  # Define inference op
  sparse_index = tf.placeholder(tf.int64, [None, 2])
  sparse_ids = tf.placeholder(tf.int64, [None])
  sparse_values = tf.placeholder(tf.float32, [None])
  sparse_shape = tf.placeholder(tf.int64, [2])
  inference_ids = tf.SparseTensor(sparse_index, sparse_ids, sparse_shape)
  inference_values = tf.SparseTensor(sparse_index, sparse_values, sparse_shape)
  inference_logits = inference(inference_ids, inference_values, False)
  #inference_softmax = tf.nn.softmax(inference_logits) # regression
  #inference_op = tf.argmax(inference_softmax, 1)
  inference_op = inference_logits
  keys_placeholder = tf.placeholder(tf.int32, shape=[None, 1])
  keys = tf.identity(keys_placeholder)
  model_signature = {
      "inputs":
      exporter.generic_signature({
          "keys": keys_placeholder,
          "indexs": sparse_index,
          "ids": sparse_ids,
          "values": sparse_values,
          "shape": sparse_shape
      }),
      "outputs":
      exporter.generic_signature({
          "keys": keys,
          #"softmax": inference_softmax,
          "prediction": inference_op
      })
  }

  # Initialize saver and summary
  saver = tf.train.Saver()
  tf.summary.scalar("loss", loss)
  tf.summary.scalar("train_accuracy", train_accuracy)
  #tf.summary.scalar("train_auc", train_auc)
  tf.summary.scalar("validate_accuracy", validate_accuracy)
  #tf.summary.scalar("validate_auc", validate_auc)
  summary_op = tf.summary.merge_all()
  init_op = [
      tf.global_variables_initializer(),
      tf.local_variables_initializer()
  ]

  # Create session to run
  with tf.Session() as sess:
    logging.info("Start to run with mode: {}".format(MODE))
    writer = tf.summary.FileWriter(OUTPUT_PATH, sess.graph)
    sess.run(init_op)
    if MODE == "train":
      # Restore session and start queue runner
      restore_session_from_checkpoint(sess, saver, LATEST_CHECKPOINT)
      coord = tf.train.Coordinator()
      threads = tf.train.start_queue_runners(coord=coord, sess=sess)
      start_time = datetime.datetime.now()

      try:
        while not coord.should_stop():
          if FLAGS.benchmark_mode:
            sess.run(train_op)
          else:
            _, sample_size, step, tloss, tlabels, tlogit, tmae  = sess.run([
                train_op, sample_inc_op,  global_step, loss, batch_labels, logits, cross_entropy
            ])
            logging.info("Step: {}, sample_size: {}, train loss: {}, mae: {}, batch_labels: {}, tpred: {}".format(
                step, sample_size, tloss, tmae, tlabels[:1], tlogit[:1]))

            # Print state while training, 个人理解：此处用模型进行一次预测
            if step % FLAGS.steps_to_validate == 0:
              loss_value, train_accuracy_value, validate_accuracy_value, summary_value, validate_size = sess.run(
                  [
                      loss, train_accuracy, validate_accuracy, summary_op, validate_inc_op
                  ])
              end_time = datetime.datetime.now()

              logging.info(
                "[{}] Step: {}, validate_size: {} loss: {}, train_acc: {},  valid_acc: {}".
                  format(end_time - start_time, step, validate_size, loss_value,
                         train_accuracy_value, validate_accuracy_value))
              writer.add_summary(summary_value, step)
              saver.save(sess, CHECKPOINT_FILE, global_step=step)
              start_time = end_time
      except tf.errors.OutOfRangeError as e:
        logging.info("Step: {} catch OutOfRangeErr: {}".format(step, e))
        if FLAGS.benchmark_mode:
          print("Finish training for benchmark")
          exit(0)
        else:
          # Export the model after training
          export_model(sess, saver, model_signature, FLAGS.model_path,
                       FLAGS.model_version)
      finally:
        coord.request_stop()
      coord.join(threads)

    elif MODE == "export":
      if not restore_session_from_checkpoint(sess, saver, LATEST_CHECKPOINT):
        logging.error("No checkpoint found, exit now")
        exit(1)

      # Export the model
      export_model(sess, saver, model_signature, FLAGS.model_path,
                   FLAGS.model_version)

    elif MODE == "savedmodel":
      if not restore_session_from_checkpoint(sess, saver, LATEST_CHECKPOINT):
        logging.error("No checkpoint found, exit now")
        exit(1)

      logging.info(
          "Export the saved model to {}".format(FLAGS.saved_model_path))
      export_path_base = FLAGS.saved_model_path
      export_path = os.path.join(
          compat.as_bytes(export_path_base),
          compat.as_bytes(str(FLAGS.model_version)))

      model_signature = signature_def_utils.build_signature_def(
          inputs={
              "keys": utils.build_tensor_info(keys_placeholder),
              "indexs": utils.build_tensor_info(sparse_index),
              "ids": utils.build_tensor_info(sparse_ids),
              "values": utils.build_tensor_info(sparse_values),
              "shape": utils.build_tensor_info(sparse_shape)
          },
          outputs={
              "keys": utils.build_tensor_info(keys),
              #"softmax": utils.build_tensor_info(inference_softmax),
              "prediction": utils.build_tensor_info(inference_op)
          },
          method_name=signature_constants.PREDICT_METHOD_NAME)

      try:
        builder = saved_model_builder.SavedModelBuilder(export_path)
        builder.add_meta_graph_and_variables(
            sess,
            [tag_constants.SERVING],
            clear_devices=True,
            signature_def_map={
                signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY:
                model_signature,
            },
            #legacy_init_op=legacy_init_op)
            legacy_init_op=tf.group(
                tf.initialize_all_tables(), name="legacy_init_op"))

        builder.save()
      except Exception as e:
        logging.error("Fail to export saved model, exception: {}".format(e))

    elif MODE == "inference":
      if not restore_session_from_checkpoint(sess, saver, LATEST_CHECKPOINT):
        logging.error("No checkpoint found, exit now")
        exit(1)

      # Load inference test data
      #inference_result_file_name = "./inference_result.txt"    # regression
      #inference_test_file_name = "./data/a8a_test.libsvm"
      inference_result_file_name = FLAGS.inference_test_file
      inference_test_file_name = FLAGS.inference_result_file
      labels = []
      feature_ids = []
      feature_values = []
      feature_index = []
      ins_num = 0
      for line in open(inference_test_file_name, "r"):
        tokens = line.split(" ")
        labels.append(int(tokens[0]))
        feature_num = 0
        for feature in tokens[1:]:
          feature_id, feature_value = feature.split(":")
          feature_ids.append(int(feature_id))
          feature_values.append(float(feature_value))
          feature_index.append([ins_num, feature_num])
          feature_num += 1
        ins_num += 1

      # Run inference
      start_time = datetime.datetime.now()
      #prediction, prediction_softmax = sess.run(   # regression
      #    [inference_op, inference_softmax],
      prediction = sess.run([inference_op],
          feed_dict={
              sparse_index: feature_index,
              sparse_ids: feature_ids,
              sparse_values: feature_values,
              sparse_shape: [ins_num, FEATURE_SIZE]
          })

      end_time = datetime.datetime.now()

      # Compute accuracy
      label_number = len(labels)
      correct_label_number = 0
      for i in range(label_number):
        if labels[i] == prediction[i]:
          correct_label_number += 1
      accuracy = float(correct_label_number) / label_number

      # Compute auc
      #expected_labels = np.array(labels)
      #predict_labels = prediction_softmax[:, 0] # regression
      #fpr, tpr, thresholds = metrics.roc_curve(
      #    expected_labels, predict_labels, pos_label=0)
      #auc = metrics.auc(fpr, tpr)
      #logging.info("[{}] Inference accuracy: {}, auc: {}".format(
      #    end_time - start_time, accuracy, auc))

      # Save result into the file
      #np.savetxt(inference_result_file_name, prediction_softmax, delimiter=",")
      np.savetxt(inference_result_file_name, inference_op, delimiter="\t")    # regression
      logging.info(
          "Save result to file: {}".format(inference_result_file_name))

    elif MODE == "inference_with_tfrecords":
      if not restore_session_from_checkpoint(sess, saver, LATEST_CHECKPOINT):
        logging.error("No checkpoint found, exit now")
        exit(1)

      # Load inference test data
      inference_result_file_name = "./inference_result.txt"
      inference_test_file_name = "./data/a8a/a8a_test.libsvm.tfrecords"
      #inference_test_file_name = "hdfs://namenode:8020/user/tobe/deep_recommend_system/data/a8a/a8a_test.libsvm.tfrecords"

      # batch_labels = features["label"]
      # batch_ids = features["ids"]
      # batch_values = features["values"]
      batch_feature_index = []
      batch_labels = []
      batch_ids = []
      batch_values = []
      ins_num = 0

      # Read from TFRecords files
      for serialized_example in tf.python_io.tf_record_iterator(
          inference_test_file_name):
        # Get serialized example from file
        example = tf.train.Example()
        example.ParseFromString(serialized_example)
        label = example.features.feature["label"].float_list.value
        ids = example.features.feature["ids"].int64_list.value
        values = example.features.feature["values"].float_list.value
        #print("label: {}, features: {}".format(label, " ".join([str(id) + ":" + str(value) for id, value in zip(ids, values)])))
        batch_labels.append(label)
        # Notice that using extend() instead of append() to flatten the values
        batch_ids.extend(ids)
        batch_values.extend(values)
        for i in xrange(len(ids)):
          batch_feature_index.append([ins_num, i])

        ins_num += 1

      # Run inference
      start_time = datetime.datetime.now()
      #prediction, prediction_softmax = sess.run(   # regression
      #    [inference_op, inference_softmax],
      prediction = sess.run([inference_op],
          feed_dict={
              sparse_index: batch_feature_index,
              sparse_ids: batch_ids,
              sparse_values: batch_values,
              sparse_shape: [ins_num, FEATURE_SIZE]
          })

      end_time = datetime.datetime.now()

      # Compute accuracy
      label_number = len(batch_labels)
      correct_label_number = 0
      for i in range(label_number):
        correct_label_number += math_ops.abs(batch_labels[i] - prediction[i])
      accuracy = float(correct_label_number) / label_number

      ## Compute auc
      #expected_labels = np.array(batch_labels)
      #predict_labels = prediction_softmax[:, 0]
      #fpr, tpr, thresholds = metrics.roc_curve(
      #    expected_labels, predict_labels, pos_label=0)
      #auc = metrics.auc(fpr, tpr)
      #logging.info("[{}] Inference accuracy: {}, auc: {}".format(
      #    end_time - start_time, accuracy, auc))

      # Save result into the file
      #np.savetxt(inference_result_file_name, prediction_softmax, delimiter=",")
      np.savetxt(inference_result_file_name, inference_op, delimiter="\t")
      logging.info(
          "Save result to file: {}".format(inference_result_file_name))


def get_optimizer(optimizer, learning_rate):
  logging.info("Use the optimizer: {}".format(optimizer))
  if optimizer == "sgd":
    return tf.train.GradientDescentOptimizer(learning_rate)
  elif optimizer == "adadelta":
    return tf.train.AdadeltaOptimizer(learning_rate)
  elif optimizer == "adagrad":
    return tf.train.AdagradOptimizer(learning_rate)
  elif optimizer == "adam":
    return tf.train.AdamOptimizer(learning_rate)
  elif optimizer == "ftrl":
    return tf.train.FtrlOptimizer(learning_rate)
  elif optimizer == "rmsprop":
    return tf.train.RMSPropOptimizer(learning_rate)
  else:
    logging.error("Unknow optimizer: {}, exit".format(optimizer))
    exit(1)


def restore_session_from_checkpoint(sess, saver, checkpoint):
  if checkpoint:
    logging.info("Restore session from checkpoint: {}".format(checkpoint))
    saver.restore(sess, checkpoint)
    return True
  else:
    return False


def export_model(sess, saver, signature, model_path, model_version):
  logging.info("Export the model to {}".format(model_path))
  model_exporter = exporter.Exporter(saver)
  model_exporter.init(
      sess.graph.as_graph_def(),
      named_graph_signatures=signature,
      clear_devices=True)
  try:
    model_exporter.export(model_path, tf.constant(model_version), sess)
  except Exception as e:
    logging.error("Fail to export model, exception: {}".format(e))


if __name__ == "__main__":
  main()
