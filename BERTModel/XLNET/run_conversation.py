from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from os.path import join
from absl import flags
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
rootPath=os.path.split(rootPath)[0]
sys.path.append(rootPath)
import csv
import collections
import numpy as np
import time
import math
import json
import random
from copy import copy
from collections import defaultdict as dd

import absl.logging as _logging  # pylint: disable=unused-import
import tensorflow as tf

import sentencepiece as spm
from evaluation_tool import DoubanMetrics
from BERTModel.XLNET.data_utils import SEP_ID, VOCAB_SIZE, CLS_ID
from BERTModel.XLNET import model_utils
from BERTModel.XLNET import function_builder
from BERTModel.XLNET.classifier_utils import PaddingInputExample
from BERTModel.XLNET.classifier_utils import convert_single_example
from BERTModel.XLNET.prepro_utils import preprocess_text, encode_ids


# Model
flags.DEFINE_string("model_config_path", default=None,help="Model config path.")
flags.DEFINE_float("dropout", default=0.1,help="Dropout rate.")
flags.DEFINE_float("dropatt", default=0.1,help="Attention dropout rate.")
flags.DEFINE_integer("clamp_len", default=-1,help="Clamp length")
flags.DEFINE_string("summary_type", default="last",help="Method used to summarize a sequence into a compact vector.")
flags.DEFINE_bool("use_summ_proj", default=True,help="Whether to use projection for summarizing sequences.")
flags.DEFINE_bool("use_bfloat16", False,help="Whether to use bfloat16.")

# Parameter initialization
flags.DEFINE_enum("init", default="normal",enum_values=["normal", "uniform"],help="Initialization method.")
flags.DEFINE_float("init_std", default=0.02,help="Initialization std when init is normal.")
flags.DEFINE_float("init_range", default=0.1,help="Initialization std when init is uniform.")

# I/O paths
flags.DEFINE_bool("overwrite_data", default=False,help="If False, will use cached data if available.")
flags.DEFINE_string("init_checkpoint", default=None,help="checkpoint path for initializing the model. Could be a pretrained model or a finetuned model.")
flags.DEFINE_string("spiece_model_file", default="",help="Sentence Piece model path.")
flags.DEFINE_string("data_dir", default="",help="Directory for input data.")

# TPUs and machines
flags.DEFINE_bool("use_tpu", default=False, help="whether to use TPU.")
flags.DEFINE_integer("num_hosts", default=1, help="How many TPU hosts.")
flags.DEFINE_integer("num_core_per_host", default=8,help="8 for TPU v2 and v3-8, 16 for larger TPU v3 pod. In the context of GPU training, it refers to the number of GPUs used.")
flags.DEFINE_string("tpu_job_name", default=None, help="TPU worker job name.")
flags.DEFINE_string("tpu", default=None, help="TPU name.")
flags.DEFINE_string("tpu_zone", default=None, help="TPU zone.")
flags.DEFINE_string("gcp_project", default=None, help="gcp project.")
flags.DEFINE_string("master", default=None, help="master")
flags.DEFINE_integer("iterations", default=10,help="number of iterations per TPU training loop.")
flags.DEFINE_integer("save_steps", default=10,help="Save the model for every save_steps. If None, not to save any model.")
################################################
flags.DEFINE_string("record_dir", "recordset","The tf_record directory for difference corpus")
flags.DEFINE_string("corpus", "douban","The tf_record directory for difference corpus")
flags.DEFINE_string("model", "chbert","The tf_record directory for difference corpus")
flags.DEFINE_string("gpu", "0","the gpu device should be used")
# training
flags.DEFINE_bool("do_prepare", False, "Whether to run training.")
flags.DEFINE_bool("do_train", default=False, help="whether to do training")
flags.DEFINE_integer("warmup_steps", default=0, help="number of warmup steps")
flags.DEFINE_float("learning_rate", default=1e-5, help="initial learning rate")
flags.DEFINE_float("lr_layer_decay_rate", 1.0,help="Top layer: lr[L] = FLAGS.learning_rate.Low layer: lr[l-1] = lr[l] * lr_layer_decay_rate.")
flags.DEFINE_float("min_lr_ratio", default=0.0,help="min lr ratio for cos decay.")
flags.DEFINE_float("clip", default=1.0, help="Gradient clipping")
flags.DEFINE_integer("max_save", default=0,help="Max number of checkpoints to save. Use 0 to save all.")
flags.DEFINE_integer("train_batch_size", default=8,help="Batch size for training")
flags.DEFINE_float("weight_decay", default=0.00, help="Weight decay rate")
flags.DEFINE_float("adam_epsilon", default=1e-8, help="Adam epsilon")
flags.DEFINE_string("decay_method", default="poly", help="poly or cos")
# evaluation
flags.DEFINE_bool("do_eval", default=False, help="whether to do eval")
flags.DEFINE_bool("do_predict", default=False, help="whether to do prediction")
flags.DEFINE_float("predict_threshold", default=0,help="Threshold for binary prediction.")
flags.DEFINE_string("eval_split", default="dev", help="could be dev or test")
flags.DEFINE_integer("eval_batch_size", default=128,help="batch size for evaluation")
flags.DEFINE_integer("predict_batch_size", default=128,help="batch size for prediction.")
flags.DEFINE_string("predict_dir", default=None,help="Dir for saving prediction files.")
flags.DEFINE_bool("eval_all_ckpt", default=False,help="Eval all ckpts. If False, only evaluate the last one.")
flags.DEFINE_string("predict_ckpt", default=None,help="Ckpt path for do_predict. If None, use the last one.")
# task specific
flags.DEFINE_float("warmup_proportion", 0,"Proportion of training to perform linear learning rate warmup for. E.g., 0.1 = 10% of training.")
flags.DEFINE_float("num_train_epochs", 3.0,"Total number of training epochs to perform.")
flags.DEFINE_string("output_dir", default=None, help="output_dir")
flags.DEFINE_string("task_name", default=None, help="Task name")
flags.DEFINE_integer("max_seq_length", default=128, help="Max sequence length")
flags.DEFINE_integer("shuffle_buffer", default=2048,help="Buffer size used for shuffle.")
flags.DEFINE_integer("num_passes", default=1,help="Num passes for processing training data. This is use to batch data without loss for TPUs.")
flags.DEFINE_bool("uncased", default=False,help="Use uncased.")
flags.DEFINE_string("cls_scope", default=None,help="Classifier layer scope.")
flags.DEFINE_bool("is_regression", default=False,help="Whether it's a regression task.")
FLAGS = flags.FLAGS


class InputExample(object):
  """A single training/test example for simple sequence classification."""

  def __init__(self, guid, text_a, text_b=None, label=None):
    """Constructs a InputExample.
    Args:
      guid: Unique id for the example.
      text_a: string. The untokenized text of the first sequence. For single
        sequence tasks, only this sequence must be specified.
      text_b: (Optional) string. The untokenized text of the second sequence.
        Only must be specified for sequence pair tasks.
      label: (Optional) string. The label of the example. This should be
        specified for train and dev examples, but not for test examples.
    """
    self.guid = guid
    self.text_a = text_a
    self.text_b = text_b
    self.label = label


class DataProcessor(object):
  """Base class for data converters for sequence classification data sets."""

  def get_train_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the train set."""
    raise NotImplementedError()

  def get_dev_examples(self, data_dir):
    """Gets a collection of `InputExample`s for the dev set."""
    raise NotImplementedError()

  def get_test_examples(self, data_dir):
    """Gets a collection of `InputExample`s for prediction."""
    raise NotImplementedError()

  def get_labels(self):
    """Gets the list of labels for this data set."""
    raise NotImplementedError()

  @classmethod
  def _read_tsv(cls, input_file, quotechar=None):
    """Reads a tab separated value file."""
    with tf.gfile.Open(input_file, "r") as f:
      reader = csv.reader(f, delimiter="\t", quotechar=quotechar)
      lines = []
      for line in reader:
        if len(line) == 0: continue
        lines.append(line)
      return lines
  def _read_txt(self,input_file):
    with tf.gfile.Open(input_file, "r") as f:
        for line in f:
            line=line.rstrip()
            yield line

class ConverProcessor(DataProcessor):
    def get_train_examples(self, data_dir,corpus):
        return self._create_examples(self._read_txt(os.path.join(data_dir, corpus, "train.txt")), "train")
    def get_dev_examples(self, data_dir,corpus):
        return self._create_examples(self._read_txt(os.path.join(data_dir, corpus, "test.txt")), "dev")
    def get_test_examples(self, data_dir,corpus):
        return self._create_examples(self._read_txt(os.path.join(data_dir, corpus, "test.txt")), "test")
    
    def get_labels(self):
        return ["0", "1"]
    def _create_examples(self, lines, set_type): 
        examples = []
        for i,line in enumerate(lines):
            guid=str(i)
            contents=line.split("\t")
            label=str(contents[0])
            history=contents[1:-1]
            response=contents[-1:]
            history=" ".join(history)
            response=" ".join(response)
            text_a = history
            text_b = response
            label = label
            examples.append(InputExample(guid=guid, text_a=text_a, text_b=text_b, label=label))
        return examples

class Yelp5Processor(DataProcessor):
  def get_train_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "train.csv"))

  def get_dev_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "test.csv"))

  def get_labels(self):
    """See base class."""
    return ["1", "2", "3", "4", "5"]

  def _create_examples(self, input_file):
    """Creates examples for the training and dev sets."""
    examples = []
    with tf.gfile.Open(input_file) as f:
      reader = csv.reader(f)
      for i, line in enumerate(reader):

        label = line[0]
        text_a = line[1].replace('""', '"').replace('\\"', '"')
        examples.append(
            InputExample(guid=str(i), text_a=text_a, text_b=None, label=label))
    return examples


class ImdbProcessor(DataProcessor):
  def get_labels(self):
    return ["neg", "pos"]

  def get_train_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "train"))

  def get_dev_examples(self, data_dir):
    return self._create_examples(os.path.join(data_dir, "test"))

  def _create_examples(self, data_dir):
    examples = []
    for label in ["neg", "pos"]:
      cur_dir = os.path.join(data_dir, label)
      for filename in tf.gfile.ListDirectory(cur_dir):
        if not filename.endswith("txt"): continue

        path = os.path.join(cur_dir, filename)
        with tf.gfile.Open(path) as f:
          text = f.read().strip().replace("<br />", " ")
        examples.append(InputExample(
            guid="unused_id", text_a=text, text_b=None, label=label))
    return examples



def file_based_convert_examples_to_features(
    examples, label_list, max_seq_length, tokenize_fn, output_file,
    num_passes=1):
  """Convert a set of `InputExample`s to a TFRecord file."""

  # do not create duplicated records
  if tf.gfile.Exists(output_file) and not FLAGS.overwrite_data:
    tf.logging.info("Do not overwrite tfrecord {} exists.".format(output_file))
    return

  tf.logging.info("Create new tfrecord {}.".format(output_file))

  writer = tf.python_io.TFRecordWriter(output_file)

  if num_passes > 1:
    examples *= num_passes

  for (ex_index, example) in enumerate(examples):
    if ex_index % 10000 == 0:
      tf.logging.info("Writing example {} of {}".format(ex_index,
                                                        len(examples)))

    feature = convert_single_example(ex_index, example, label_list,max_seq_length, tokenize_fn)

    def create_int_feature(values):
      f = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
      return f

    def create_float_feature(values):
      f = tf.train.Feature(float_list=tf.train.FloatList(value=list(values)))
      return f

    features = collections.OrderedDict()
    features["input_ids"] = create_int_feature(feature.input_ids)
    features["input_mask"] = create_float_feature(feature.input_mask)
    features["segment_ids"] = create_int_feature(feature.segment_ids)
    if label_list is not None:
      features["label_ids"] = create_int_feature([feature.label_id])
    else:
      features["label_ids"] = create_float_feature([float(feature.label_id)])
    features["is_real_example"] = create_int_feature(
        [int(feature.is_real_example)])

    tf_example = tf.train.Example(features=tf.train.Features(feature=features))
    writer.write(tf_example.SerializeToString())
  writer.close()


def file_based_input_fn_builder(input_file, seq_length, is_training,
                                drop_remainder):
  """Creates an `input_fn` closure to be passed to TPUEstimator."""


  name_to_features = {
      "input_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "input_mask": tf.FixedLenFeature([seq_length], tf.float32),
      "segment_ids": tf.FixedLenFeature([seq_length], tf.int64),
      "label_ids": tf.FixedLenFeature([], tf.int64),
      "is_real_example": tf.FixedLenFeature([], tf.int64),
  }
  if FLAGS.is_regression:
    name_to_features["label_ids"] = tf.FixedLenFeature([], tf.float32)

  tf.logging.info("Input tfrecord file {}".format(input_file))

  def _decode_record(record, name_to_features):
    """Decodes a record to a TensorFlow example."""
    example = tf.parse_single_example(record, name_to_features)

    # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
    # So cast all int64 to int32.
    for name in list(example.keys()):
      t = example[name]
      if t.dtype == tf.int64:
        t = tf.cast(t, tf.int32)
      example[name] = t

    return example

  def input_fn(params, input_context=None):
    """The actual input function."""
    if FLAGS.use_tpu:
      batch_size = params["batch_size"]
    elif is_training:
      batch_size = FLAGS.train_batch_size
    elif FLAGS.do_eval:
      batch_size = FLAGS.eval_batch_size
    else:
      batch_size = FLAGS.predict_batch_size

    d = tf.data.TFRecordDataset(input_file)
    # Shard the dataset to difference devices
    if input_context is not None:
      tf.logging.info("Input pipeline id %d out of %d",
          input_context.input_pipeline_id, input_context.num_replicas_in_sync)
      d = d.shard(input_context.num_input_pipelines,
                  input_context.input_pipeline_id)

    # For training, we want a lot of parallel reading and shuffling.
    # For eval, we want no shuffling and parallel reading doesn't matter.
    if is_training:
      d = d.shuffle(buffer_size=FLAGS.shuffle_buffer)
      d = d.repeat()

    d = d.apply(
        tf.contrib.data.map_and_batch(
            lambda record: _decode_record(record, name_to_features),
            batch_size=batch_size,
            drop_remainder=drop_remainder))

    return d

  return input_fn


def get_model_fn(n_class):
  def model_fn(features, labels, mode, params):
    #### Training or Evaluation
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)

    #### Get loss from inputs
    if FLAGS.is_regression:
      (total_loss, per_example_loss, logits
          ) = function_builder.get_regression_loss(FLAGS, features, is_training)
    else:
      (total_loss, per_example_loss, logits
          ) = function_builder.get_classification_loss(
          FLAGS, features, n_class, is_training)

    #### Check model parameters
    num_params = sum([np.prod(v.shape) for v in tf.trainable_variables()])
    tf.logging.info('#params: {}'.format(num_params))

    #### load pretrained models
    scaffold_fn = model_utils.init_from_checkpoint(FLAGS)

    #### Evaluation mode
    if mode == tf.estimator.ModeKeys.EVAL:
      assert FLAGS.num_hosts == 1

      def metric_fn(per_example_loss, label_ids, logits, is_real_example):
        predictions = tf.argmax(logits, axis=-1, output_type=tf.int32)
        eval_input_dict = {
            'labels': label_ids,
            'predictions': predictions,
            'weights': is_real_example
        }
        accuracy = tf.metrics.accuracy(**eval_input_dict)

        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        return {
            'eval_accuracy': accuracy,
            'eval_loss': loss}

      def regression_metric_fn(
          per_example_loss, label_ids, logits, is_real_example):
        loss = tf.metrics.mean(values=per_example_loss, weights=is_real_example)
        pearsonr = tf.contrib.metrics.streaming_pearson_correlation(
            logits, label_ids, weights=is_real_example)
        return {'eval_loss': loss, 'eval_pearsonr': pearsonr}

      is_real_example = tf.cast(features["is_real_example"], dtype=tf.float32)

      #### Constucting evaluation TPUEstimatorSpec with new cache.
      label_ids = tf.reshape(features['label_ids'], [-1])

      if FLAGS.is_regression:
        metric_fn = regression_metric_fn
      else:
        metric_fn = metric_fn
      metric_args = [per_example_loss, label_ids, logits, is_real_example]

      if FLAGS.use_tpu:
        eval_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metrics=(metric_fn, metric_args),
            scaffold_fn=scaffold_fn)
      else:
        eval_spec = tf.estimator.EstimatorSpec(
            mode=mode,
            loss=total_loss,
            eval_metric_ops=metric_fn(*metric_args))

      return eval_spec

    elif mode == tf.estimator.ModeKeys.PREDICT:
      label_ids = tf.reshape(features["label_ids"], [-1])

      predictions = {
          "probabilities": logits,
          "labels": label_ids,
      }

      if FLAGS.use_tpu:
        output_spec = tf.contrib.tpu.TPUEstimatorSpec(
            mode=mode, predictions=predictions, scaffold_fn=scaffold_fn)
      else:
        output_spec = tf.estimator.EstimatorSpec(
            mode=mode, predictions=predictions)
      return output_spec

    #### Configuring the optimizer
    train_op, learning_rate, _ = model_utils.get_train_op(FLAGS, total_loss)

    monitor_dict = {}
    monitor_dict["lr"] = learning_rate

    #### Constucting training TPUEstimatorSpec with new cache.
    if FLAGS.use_tpu:
      #### Creating host calls
      if not FLAGS.is_regression:
        label_ids = tf.reshape(features['label_ids'], [-1])
        predictions = tf.argmax(logits, axis=-1, output_type=label_ids.dtype)
        is_correct = tf.equal(predictions, label_ids)
        accuracy = tf.reduce_mean(tf.cast(is_correct, tf.float32))

        monitor_dict["accuracy"] = accuracy

        host_call = function_builder.construct_scalar_host_call(
            monitor_dict=monitor_dict,
            model_dir=FLAGS.model_dir,
            prefix="train/",
            reduce_fn=tf.reduce_mean)
      else:
        host_call = None

      train_spec = tf.contrib.tpu.TPUEstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op, host_call=host_call,
          scaffold_fn=scaffold_fn)
    else:
      train_spec = tf.estimator.EstimatorSpec(
          mode=mode, loss=total_loss, train_op=train_op)

    return train_spec

  return model_fn
def path_verify():
    path1=os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model)
    path2=os.path.join(FLAGS.output_dir,FLAGS.corpus,FLAGS.model)
    tf.gfile.MakeDirs(path1)
    tf.gfile.MakeDirs(path2)

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  #### Validate flags
  if FLAGS.save_steps is not None:
    FLAGS.iterations = min(FLAGS.iterations, FLAGS.save_steps)
  path_verify()
  processors = {
      'imdb': ImdbProcessor,
      "yelp5": Yelp5Processor,
      "conv": ConverProcessor,
  }

  if not FLAGS.do_train and not FLAGS.do_eval and not FLAGS.do_predict and not FLAGS.do_prepare:
    raise ValueError(
        "At least one of `do_train`, `do_eval, `do_predict` or `do_prepare` must be True.")

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name]()
  label_list = processor.get_labels() if not FLAGS.is_regression else None

  sp = spm.SentencePieceProcessor()
  sp.Load(FLAGS.spiece_model_file)
  def tokenize_fn(text):
    text = preprocess_text(text, lower=FLAGS.uncased)
    return encode_ids(sp, text)
  run_config = model_utils.configure_tpu(FLAGS)
  model_fn = get_model_fn(len(label_list))
  if FLAGS.use_tpu:
    estimator = tf.contrib.tpu.TPUEstimator(
        use_tpu=FLAGS.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=FLAGS.train_batch_size,
        predict_batch_size=FLAGS.predict_batch_size,
        eval_batch_size=FLAGS.eval_batch_size)
  else:
    estimator = tf.estimator.Estimator(
        model_fn=model_fn,
        config=run_config)
  if FLAGS.do_prepare:
    meta_json=dict()
    meta_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"meta.json")
    train_examples = processor.get_train_examples(FLAGS.data_dir,FLAGS.corpus)
    np.random.shuffle(train_examples)
    num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    meta_json["num_train_examples"]=len(train_examples)
    meta_json["num_train_steps"]=num_train_steps
    meta_json["num_warmup_steps"]=num_warmup_steps
    train_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"train.tf_record")
    file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenize_fn,train_file, FLAGS.num_passes)
    eval_examples = processor.get_dev_examples(FLAGS.data_dir,FLAGS.corpus)
    eval_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"eval.tf_record")
    file_based_convert_examples_to_features(eval_examples, label_list, FLAGS.max_seq_length, tokenize_fn,eval_file)
    meta_json["num_actual_eval_examples"]=len(eval_examples)
    predict_examples = processor.get_test_examples(FLAGS.data_dir,FLAGS.corpus)
    meta_json["num_actual_predict_examples"]=len(predict_examples)
    predict_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"predict.tf_record")
    file_based_convert_examples_to_features(predict_examples, label_list, FLAGS.max_seq_length, tokenize_fn,predict_file)
    with open(meta_file,"w",encoding="utf-8") as f:
        f.write(json.dumps(meta_json, ensure_ascii=False,indent=4))
    print("prepare done .......")
    return 
  meta_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"meta.json")
  with open(meta_file,"r",encoding="utf-8") as f:
    meta_json=json.load(f)
  model_dir=os.path.join(FLAGS.output_dir,FLAGS.corpus,FLAGS.model) 
  if FLAGS.do_train:
    num_train_steps = meta_json["num_train_steps"]
    train_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"train.tf_record")
    tf.logging.info("Use tfrecord file {}".format(train_file))
    train_input_fn = file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)
  if FLAGS.do_eval:
    num_actual_eval_examples = meta_json["num_actual_eval_examples"]
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)
    eval_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model, "eval.tf_record")
    eval_steps = None
    eval_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)

    ####
    best_trial_info_file = os.path.join(model_dir, "best_trial.txt")
    def _best_trial_info():
      """Returns information about which checkpoints have been evaled so far."""
      if tf.gfile.Exists(best_trial_info_file):
        with tf.gfile.GFile(best_trial_info_file, "r") as best_info:
          global_step, best_metric_global_step, metric_value = (
              best_info.read().split(":"))
          global_step = int(global_step)
          best_metric_global_step = int(best_metric_global_step)
          metric_value = float(metric_value)
      else:
        metric_value = -1
        best_metric_global_step = -1
        global_step = -1
      tf.logging.info(
          "Best trial info: Step: %s, Best Value Step: %s, "
          "Best Value: %s", global_step, best_metric_global_step, metric_value)
      return global_step, best_metric_global_step, metric_value
    def _remove_checkpoint(checkpoint_path):
      for ext in ["meta", "data-00000-of-00001", "index"]:
        src_ckpt = checkpoint_path + ".{}".format(ext)
        tf.logging.info("removing {}".format(src_ckpt))
        tf.gfile.Remove(src_ckpt)
    def _find_valid_cands(curr_step):
      filenames = tf.gfile.ListDirectory(model_dir)
      candidates = []
      for filename in filenames:
        if filename.endswith(".index"):
          ckpt_name = filename[:-6]
          idx = ckpt_name.split("-")[-1]
          if not isinstance(idx, int):
            continue
          if int(idx) > curr_step:
            candidates.append(filename)
      return candidates
    output_eval_file = os.path.join(FLAGS.output_dir, FLAGS.corpus,FLAGS.model,"eval_results.txt")
    key_name = "R10@1"
    global_step, best_perf_global_step, best_perf = _best_trial_info()
    writer = tf.gfile.GFile(output_eval_file, "w")
    steps_and_files = {}
    filenames = tf.gfile.ListDirectory(model_dir)
    for filename in filenames:
        if filename.endswith(".index"):
            ckpt_name = filename[:-6]
            cur_filename = os.path.join(model_dir, ckpt_name)
            if cur_filename.split("-")[-1] == "best":
                continue
            gstep = int(cur_filename.split("-")[-1])
            if gstep not in steps_and_files:
                tf.logging.info("Add {} to eval list.".format(cur_filename))
                steps_and_files[gstep] = cur_filename
    tf.logging.info("found {} files.".format(len(steps_and_files)))
    if not steps_and_files:
        tf.logging.info("found 0 file")
    for checkpoint in sorted(steps_and_files.items()):
        step, checkpoint_path = checkpoint
        result = estimator.predict(input_fn=eval_input_fn,checkpoint_path=checkpoint_path)
        score,label_list=[],[]
        for i,prediction in enumerate(result):
            probabilities = prediction["probabilities"].tolist()
            label=prediction["labels"].tolist()
            score.append(probabilities[-1])
            label_list.append(label)
        result_metrics=DoubanMetrics(score,label_list,count=10)
        global_step=step
        if not isinstance(global_step, int):
            continue
        tf.logging.info("***** Eval results *****")
        for key in sorted(result_metrics.keys()):
            tf.logging.info("  %s = %s", key, str(result_metrics[key]))
            writer.write("%s = %s\n" % (key, str(result_metrics[key])))
        writer.write("best = {}\n".format(best_perf))
        if result_metrics[key_name] >=best_perf:
            best_perf = result_metrics[key_name]
            best_perf_global_step = global_step
            for ext in ["meta", "data-00000-of-00001", "index"]:
                src_ckpt = "model.ckpt-{}.{}".format(best_perf_global_step, ext)
                tgt_ckpt = "model.ckpt-best.{}".format(ext)
                tf.logging.info("saving {} to {}".format(src_ckpt, tgt_ckpt))
                tf.gfile.Rename(os.path.join(model_dir, src_ckpt),os.path.join(model_dir, tgt_ckpt),overwrite=True)
        else:
            _remove_checkpoint(checkpoint_path)
        writer.write("=" * 50 + "\n")
        writer.flush()
        with tf.gfile.GFile(best_trial_info_file, "w") as best_info:
            best_info.write("{}:{}:{}".format(global_step, best_perf_global_step, best_perf))    
    writer.close()
  if FLAGS.do_predict:
    num_actual_predict_examples = meta_json["num_actual_predict_examples"]
    predict_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"predict.tf_record")
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d ",num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)
    predict_input_fn = file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=False)
    checkpoint_path = os.path.join(model_dir, "model.ckpt-best")
    result = estimator.predict(input_fn=predict_input_fn,checkpoint_path=checkpoint_path)
    score,label_list=[],[]
    for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"].tolist()
        label=prediction["labels"].tolist()
        score.append(probabilities[-1])
        label_list.append(label)
    result_metrics=DoubanMetrics(score,label_list,count=10)
    print(result_metrics)
    output_predict_file = os.path.join(model_dir, "eval_results.json")
    with open(output_predict_file, "w") as writer:
        writer.write(json.dumps(result_metrics,ensure_ascii=False,indent=4))
    

if __name__ == "__main__":
  tf.app.run()
