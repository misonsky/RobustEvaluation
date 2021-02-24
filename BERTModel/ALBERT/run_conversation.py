# coding=utf-8
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import os
import sys
import json
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
rootPath=os.path.split(rootPath)[0]
sys.path.append(rootPath)
import time
from evaluation_tool import DoubanMetrics
from evaluation_tool import MutualMetrics
from evaluation_tool import groupMetrics
from BERTModel.ALBERT import classifier_utils
from BERTModel.ALBERT import fine_tuning_utils
from BERTModel.ALBERT import modeling
import tensorflow.compat.v1 as tf
from tensorflow.contrib import cluster_resolver as contrib_cluster_resolver
from tensorflow.contrib import tpu as contrib_tpu

flags = tf.flags

FLAGS = flags.FLAGS

## Required parameters
flags.DEFINE_string(
    "data_dir", None,
    "The input data dir. Should contain the .tsv files (or other data files) "
    "for the task.")

flags.DEFINE_string(
    "albert_config_file", None,
    "The config json file corresponding to the pre-trained ALBERT model. "
    "This specifies the model architecture.")
flags.DEFINE_string("record_dir", "recordset","The tf_record directory for difference corpus")
flags.DEFINE_string("corpus", "douban","The tf_record directory for difference corpus")
flags.DEFINE_string("question", "sta","The tf_record directory for difference corpus")
flags.DEFINE_string("model", "chbert","The tf_record directory for difference corpus")
flags.DEFINE_string("gpu", "0","the gpu device should be used")
flags.DEFINE_string("task_name", None, "The name of the task to train.")

flags.DEFINE_string(
    "vocab_file", None,
    "The vocabulary file that the ALBERT model was trained on.")
flags.DEFINE_bool("use_spm",False,"whether use the spm model")
flags.DEFINE_string("spm_model_file", None,"The model file for sentence piece tokenization.")

flags.DEFINE_string(
    "output_dir", None,
    "The output directory where the model checkpoints will be written.")

flags.DEFINE_string("cached_dir", None,
                    "Path to cached training and dev tfrecord file. "
                    "The file will be generated if not exist.")

## Other parameters

flags.DEFINE_string(
    "init_checkpoint", None,
    "Initial checkpoint (usually from a pre-trained BERT model).")

flags.DEFINE_string(
    "albert_hub_module_handle", None,
    "If set, the ALBERT hub module to use.")

flags.DEFINE_bool(
    "do_lower_case", True,
    "Whether to lower case the input text. Should be True for uncased "
    "models and False for cased models.")

flags.DEFINE_integer(
    "max_seq_length", 512,
    "The maximum total input sequence length after WordPiece tokenization. "
    "Sequences longer than this will be truncated, and sequences shorter "
    "than this will be padded.")

flags.DEFINE_bool("do_train", False, "Whether to run training.")
flags.DEFINE_bool("do_prepare", False, "Whether to run training.")

flags.DEFINE_bool("do_eval", False, "Whether to run eval on the dev set.")

flags.DEFINE_bool(
    "do_predict", False,
    "Whether to run the model in inference mode on the test set.")

flags.DEFINE_integer("train_batch_size", 32, "Total batch size for training.")

flags.DEFINE_integer("eval_batch_size", 8, "Total batch size for eval.")

flags.DEFINE_integer("predict_batch_size", 8, "Total batch size for predict.")

flags.DEFINE_float("learning_rate", 5e-5, "The initial learning rate for Adam.")
flags.DEFINE_float("num_train_epochs", 3.0,
                   "Total number of training epochs to perform.")

flags.DEFINE_float(
    "warmup_proportion",0.1,
    "Proportion of training to perform linear learning rate warmup for. "
    "E.g., 0.1 = 10% of training.")

flags.DEFINE_integer("save_checkpoints_steps", 10,
                     "How often to save the model checkpoint.")

flags.DEFINE_integer("keep_checkpoint_max", 5,
                     "How many checkpoints to keep.")

flags.DEFINE_integer("iterations_per_loop", 10,
                     "How many steps to make in each estimator call.")

flags.DEFINE_bool("use_tpu", False, "Whether to use TPU or GPU/CPU.")

flags.DEFINE_string("optimizer", "adamw", "Optimizer to use")

tf.flags.DEFINE_string(
    "tpu_name", None,
    "The Cloud TPU to use for training. This should be either the name "
    "used when creating the Cloud TPU, or a grpc://ip.address.of.tpu:8470 "
    "url.")

tf.flags.DEFINE_string(
    "tpu_zone", None,
    "[Optional] GCE zone where the Cloud TPU is located in. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string(
    "gcp_project", None,
    "[Optional] Project name for the Cloud TPU-enabled project. If not "
    "specified, we will attempt to automatically detect the GCE project from "
    "metadata.")

tf.flags.DEFINE_string("master", None, "[Optional] TensorFlow master URL.")

flags.DEFINE_integer(
    "num_tpu_cores", 8,
    "Only used if `use_tpu` is True. Total number of TPU cores to use.")

flags.DEFINE_string(
    "export_dir", None,
    "The directory where the exported SavedModel will be stored.")

flags.DEFINE_float(
    "threshold_to_export", float("nan"),
    "The threshold value that should be used with the exported classifier. "
    "When specified, the threshold will be attached to the exported "
    "SavedModel, and served along with the predictions. Please use the "
    "saved model cli ("
    "https://www.tensorflow.org/guide/saved_model#details_of_the_savedmodel_command_line_interface"
    ") to view the output signature of the threshold.")


def _serving_input_receiver_fn():
  """Creates an input function for serving."""
  seq_len = FLAGS.max_seq_length
  serialized_example = tf.placeholder(
      dtype=tf.string, shape=[None], name="serialized_example")
  features = {
      "input_ids": tf.FixedLenFeature([seq_len], dtype=tf.int64),
      "input_mask": tf.FixedLenFeature([seq_len], dtype=tf.int64),
      "segment_ids": tf.FixedLenFeature([seq_len], dtype=tf.int64),
  }
  feature_map = tf.parse_example(serialized_example, features=features)
  feature_map["is_real_example"] = tf.constant(1, dtype=tf.int32)
  feature_map["label_ids"] = tf.constant(0, dtype=tf.int32)

  # tf.Example only supports tf.int64, but the TPU only supports tf.int32.
  # So cast all int64 to int32.
  for name in feature_map.keys():
    t = feature_map[name]
    if t.dtype == tf.int64:
      t = tf.to_int32(t)
    feature_map[name] = t

  return tf.estimator.export.ServingInputReceiver(
      features=feature_map, receiver_tensors=serialized_example)


def _add_threshold_to_model_fn(model_fn, threshold):
  """Adds the classifier threshold to the given model_fn."""

  def new_model_fn(features, labels, mode, params):
    spec = model_fn(features, labels, mode, params)
    threshold_tensor = tf.constant(threshold, dtype=tf.float32)
    default_serving_export = spec.export_outputs[
        tf.saved_model.signature_constants.DEFAULT_SERVING_SIGNATURE_DEF_KEY]
    default_serving_export.outputs["threshold"] = threshold_tensor
    return spec

  return new_model_fn
def path_verify():
    path1=os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model)
    path2=os.path.join(FLAGS.output_dir,FLAGS.corpus,FLAGS.model)
    tf.gfile.MakeDirs(path1)
    tf.gfile.MakeDirs(path2)

def main(_):
  tf.logging.set_verbosity(tf.logging.INFO)

  processors = {
      "cola": classifier_utils.ColaProcessor,
      "mnli": classifier_utils.MnliProcessor,
      "mismnli": classifier_utils.MisMnliProcessor,
      "mrpc": classifier_utils.MrpcProcessor,
      "rte": classifier_utils.RteProcessor,
      "sst-2": classifier_utils.Sst2Processor,
      "sts-b": classifier_utils.StsbProcessor,
      "qqp": classifier_utils.QqpProcessor,
      "qnli": classifier_utils.QnliProcessor,
      "wnli": classifier_utils.WnliProcessor,
      "conv": classifier_utils.ConverProcessor,
  }

  if not (FLAGS.do_train or FLAGS.do_eval or FLAGS.do_predict or FLAGS.do_prepare):
    raise ValueError(
        "At least one of `do_train`, `do_eval`, `do_predict' or `do_prepare` "
        "must be True.")

  if not FLAGS.albert_config_file and not FLAGS.albert_hub_module_handle:
    raise ValueError("At least one of `--albert_config_file` and "
                     "`--albert_hub_module_handle` must be set")

  if FLAGS.albert_config_file:
    albert_config = modeling.AlbertConfig.from_json_file(FLAGS.albert_config_file)
    if FLAGS.max_seq_length > albert_config.max_position_embeddings:
        raise ValueError("Cannot use sequence length %d because the ALBERT model was only trained up to sequence length %d" %
          (FLAGS.max_seq_length, albert_config.max_position_embeddings))
  else:
    albert_config = None  # Get the config from TF-Hub.

  path_verify()

  task_name = FLAGS.task_name.lower()

  if task_name not in processors:
    raise ValueError("Task not found: %s" % (task_name))

  processor = processors[task_name](use_spm=FLAGS.use_spm,do_lower_case=FLAGS.do_lower_case)

  label_list = processor.get_labels()

  tokenizer = fine_tuning_utils.create_vocab(
      vocab_file=FLAGS.vocab_file,
      do_lower_case=FLAGS.do_lower_case,
      spm_model_file=FLAGS.spm_model_file,
      use_spm=FLAGS.use_spm,
      hub_module=FLAGS.albert_hub_module_handle)

  if FLAGS.do_prepare:
    meta_json=dict()
    meta_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"meta.json")
    train_examples = processor.get_train_examples(FLAGS.data_dir,FLAGS.corpus)
    num_train_steps = int(len(train_examples) / FLAGS.train_batch_size * FLAGS.num_train_epochs)
    num_warmup_steps = int(num_train_steps * FLAGS.warmup_proportion)
    meta_json["num_train_examples"]=len(train_examples)
    meta_json["num_train_steps"]=num_train_steps
    meta_json["num_warmup_steps"]=num_warmup_steps
    train_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"train.tf_record")
    classifier_utils.file_based_convert_examples_to_features(train_examples, label_list, FLAGS.max_seq_length, tokenizer,train_file, task_name)
    eval_examples = processor.get_dev_examples(FLAGS.data_dir,FLAGS.corpus)
    eval_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"eval.tf_record")
    classifier_utils.file_based_convert_examples_to_features(eval_examples, label_list, FLAGS.max_seq_length, tokenizer,eval_file, task_name)
    meta_json["num_actual_eval_examples"]=len(eval_examples)
    predict_examples = processor.get_test_examples(FLAGS.data_dir,FLAGS.corpus)
    meta_json["num_actual_predict_examples"]=len(predict_examples)
    predict_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"predict.tf_record")
    classifier_utils.file_based_convert_examples_to_features(predict_examples, label_list,FLAGS.max_seq_length, tokenizer,predict_file, task_name)
    with open(meta_file,"w",encoding="utf-8") as f:
        f.write(json.dumps(meta_json, ensure_ascii=False,indent=4))
    print("prepare done .......")
    return
  model_dir=os.path.join(FLAGS.output_dir,FLAGS.corpus,FLAGS.model)
  meta_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"meta.json")
  with open(meta_file,"r",encoding="utf-8") as f:
    meta_json=json.load(f)
  is_per_host = contrib_tpu.InputPipelineConfig.PER_HOST_V2
  if FLAGS.do_train:
    iterations_per_loop = int(min(FLAGS.iterations_per_loop,FLAGS.save_checkpoints_steps))
  else:
    iterations_per_loop = FLAGS.iterations_per_loop
  run_config = contrib_tpu.RunConfig(
      cluster=None,
      master=FLAGS.master,
      model_dir=model_dir,
      save_checkpoints_steps=int(FLAGS.save_checkpoints_steps),
      keep_checkpoint_max=0,
      tpu_config=contrib_tpu.TPUConfig(
          iterations_per_loop=iterations_per_loop,
          num_shards=FLAGS.num_tpu_cores,
          per_host_input_for_training=is_per_host))
  model_fn = classifier_utils.model_fn_builder(
      albert_config=albert_config,
      num_labels=len(label_list),
      init_checkpoint=FLAGS.init_checkpoint,
      learning_rate=FLAGS.learning_rate,
      num_train_steps=meta_json["num_train_steps"],
      num_warmup_steps=meta_json["num_warmup_steps"],
      use_tpu=FLAGS.use_tpu,
      use_one_hot_embeddings=FLAGS.use_tpu,
      task_name=task_name,
      hub_module=FLAGS.albert_hub_module_handle,
      optimizer=FLAGS.optimizer)

  if not math.isnan(FLAGS.threshold_to_export):
    model_fn = _add_threshold_to_model_fn(model_fn, FLAGS.threshold_to_export)

  # If TPU is not available, this will fall back to normal Estimator on CPU
  # or GPU.
  estimator = contrib_tpu.TPUEstimator(
      use_tpu=FLAGS.use_tpu,
      model_fn=model_fn,
      config=run_config,
      train_batch_size=FLAGS.train_batch_size,
      eval_batch_size=FLAGS.eval_batch_size,
      predict_batch_size=FLAGS.predict_batch_size,
      export_to_tpu=False)  # http://yaqs/4707241341091840

  if FLAGS.do_train:
    num_train_steps = meta_json["num_train_steps"]
    train_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,"train.tf_record")
    tf.logging.info("***** Running training *****")
    tf.logging.info("  Num examples = %d", meta_json["num_train_examples"])
    tf.logging.info("  Batch size = %d", FLAGS.train_batch_size)
    tf.logging.info("  Num steps = %d", num_train_steps)
    train_input_fn = classifier_utils.file_based_input_fn_builder(
        input_file=train_file,
        seq_length=FLAGS.max_seq_length,
        is_training=True,
        drop_remainder=True,
        task_name=task_name,
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.train_batch_size)
    estimator.train(input_fn=train_input_fn, max_steps=num_train_steps)

  if FLAGS.do_eval:
    num_actual_eval_examples = meta_json["num_actual_eval_examples"]
   
    eval_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model, "eval.tf_record")
    tf.logging.info("***** Running evaluation *****")
    tf.logging.info("  Num examples = %d", num_actual_eval_examples)
    tf.logging.info("  Batch size = %d", FLAGS.eval_batch_size)

    # This tells the estimator to run through the entire set.
    eval_steps = None
    # However, if running eval on the TPU, you will need to specify the
    # number of steps.
    eval_drop_remainder = False
    eval_input_fn = classifier_utils.file_based_input_fn_builder(
        input_file=eval_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=eval_drop_remainder,
        task_name=task_name,
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.eval_batch_size)

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
    corpus_path=os.path.join(FLAGS.corpus,FLAGS.question)
    predict_examples = processor.get_test_examples(FLAGS.data_dir,corpus_path)
    q_path=os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,FLAGS.question)
    if not os.path.exists(q_path):
        os.makedirs(q_path)
    predict_file = os.path.join(FLAGS.record_dir,FLAGS.corpus,FLAGS.model,FLAGS.question,"predict.tf_record")
    num_actual_predict_examples = len(predict_examples)
    classifier_utils.file_based_convert_examples_to_features(predict_examples, label_list,FLAGS.max_seq_length, tokenizer,predict_file, task_name)
    tf.logging.info("***** Running prediction*****")
    tf.logging.info("  Num examples = %d ",num_actual_predict_examples)
    tf.logging.info("  Batch size = %d", FLAGS.predict_batch_size)

    predict_drop_remainder = False
    predict_input_fn = classifier_utils.file_based_input_fn_builder(
        input_file=predict_file,
        seq_length=FLAGS.max_seq_length,
        is_training=False,
        drop_remainder=predict_drop_remainder,
        task_name=task_name,
        use_tpu=FLAGS.use_tpu,
        bsz=FLAGS.predict_batch_size)

    checkpoint_path = os.path.join(model_dir, "model.ckpt-best")
    result = estimator.predict(input_fn=predict_input_fn,checkpoint_path=checkpoint_path)
    score,label_list=[],[]
    for (i, prediction) in enumerate(result):
        probabilities = prediction["probabilities"].tolist()
        label=prediction["labels"].tolist()
        score.append(probabilities[-1])
        label_list.append(label)
    if FLAGS.corpus!="mutual":
        result_metrics=DoubanMetrics(score,label_list,count=10)
    else:
        result_metrics=MutualMetrics(score,label_list,count=4)
    if FLAGS.question=="div":
        if FLAGS.corpus!="mutual":
            groupMetrics(score,label_list,count=10)
        else:
            groupMetrics(score,label_list,count=4)
    print(result_metrics)
    output_predict_file = os.path.join(model_dir, "eval_results.json")
    with open(output_predict_file, "w") as writer:
        writer.write(json.dumps(result_metrics,ensure_ascii=False,indent=4))
  if FLAGS.export_dir:
    tf.gfile.MakeDirs(FLAGS.export_dir)
    checkpoint_path = os.path.join(FLAGS.output_dir, "model.ckpt-best")
    tf.logging.info("Starting to export model.")
    subfolder = estimator.export_saved_model(
        export_dir_base=FLAGS.export_dir,
        serving_input_receiver_fn=_serving_input_receiver_fn,
        checkpoint_path=checkpoint_path)
    tf.logging.info("Model exported to %s.", subfolder)


if __name__ == "__main__":
  flags.mark_flag_as_required("data_dir")
  flags.mark_flag_as_required("task_name")
  flags.mark_flag_as_required("spm_model_file")
  flags.mark_flag_as_required("vocab_file")
  flags.mark_flag_as_required("output_dir")
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = FLAGS.gpu
  tf.app.run()
