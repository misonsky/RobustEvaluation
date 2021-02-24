# coding=utf-8
# Copyright 2020 The Google Research Authors.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Fine-tunes an ELECTRA model on a downstream task."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import collections
import json
import os
import sys
curPath = os.path.abspath(os.path.dirname(__file__))
rootPath = os.path.split(curPath)[0]
rootPath=os.path.split(rootPath)[0]
sys.path.append(rootPath)
import tensorflow.compat.v1 as tf

import configure_finetuning
from finetune import preprocessing
from finetune import task_builder
from model import modeling
from model import optimization
from util import training_utils
from util import utils
from evaluation_tool import DoubanMetrics,MutualMetrics

class FinetuningModel(object):
  """Finetuning model with support for multi-task training."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
               is_training, features, num_train_steps):
    # Create a shared transformer encoder
    bert_config = training_utils.get_bert_config(config)
    self.bert_config = bert_config
    if config.debug:
      bert_config.num_hidden_layers = 3
      bert_config.hidden_size = 144
      bert_config.intermediate_size = 144 * 4
      bert_config.num_attention_heads = 4
    assert config.max_seq_length <= bert_config.max_position_embeddings
    bert_model = modeling.BertModel(
        bert_config=bert_config,
        is_training=is_training,
        input_ids=features["input_ids"],
        input_mask=features["input_mask"],
        token_type_ids=features["segment_ids"],
        use_one_hot_embeddings=config.use_tpu,
        embedding_size=config.embedding_size)
    percent_done = (tf.cast(tf.train.get_or_create_global_step(), tf.float32) /
                    tf.cast(num_train_steps, tf.float32))

    # Add specific tasks
    self.outputs = {"task_id": features["task_id"]}
    losses = []
    for task in tasks:
      with tf.variable_scope("task_specific/" + task.name):
        task_losses, task_outputs = task.get_prediction_module(
            bert_model, features, is_training, percent_done)
        losses.append(task_losses)
        self.outputs[task.name] = task_outputs
    self.loss = tf.reduce_sum(
        tf.stack(losses, -1) *
        tf.one_hot(features["task_id"], len(config.task_names)))


def model_fn_builder(config: configure_finetuning.FinetuningConfig, tasks,
                     num_train_steps, pretraining_config=None):
  """Returns `model_fn` closure for TPUEstimator."""

  def model_fn(features, labels, mode, params):
    """The `model_fn` for TPUEstimator."""
    utils.log("Building model...")
    is_training = (mode == tf.estimator.ModeKeys.TRAIN)
    model = FinetuningModel(config, tasks, is_training, features, num_train_steps)
    # Load pre-trained weights from checkpoint
    init_checkpoint = config.init_checkpoint
    if pretraining_config is not None:
      init_checkpoint = tf.train.latest_checkpoint(pretraining_config.model_dir)
      utils.log("Using checkpoint", init_checkpoint)
    tvars = tf.trainable_variables()
    scaffold_fn = None
    if init_checkpoint:
      assignment_map, _ = modeling.get_assignment_map_from_checkpoint(
          tvars, init_checkpoint)
      if config.use_tpu:
        def tpu_scaffold():
          tf.train.init_from_checkpoint(init_checkpoint, assignment_map)
          return tf.train.Scaffold()
        scaffold_fn = tpu_scaffold
      else:
        tf.train.init_from_checkpoint(init_checkpoint, assignment_map)

    # Build model for training or prediction
    if mode == tf.estimator.ModeKeys.TRAIN:
      train_op = optimization.create_optimizer(
          model.loss, config.learning_rate, num_train_steps,
          weight_decay_rate=config.weight_decay_rate,
          use_tpu=config.use_tpu,
          warmup_proportion=config.warmup_proportion,
          layerwise_lr_decay_power=config.layerwise_lr_decay,
          n_transformer_layers=model.bert_config.num_hidden_layers
      )
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          loss=model.loss,
          train_op=train_op,
          scaffold_fn=scaffold_fn,
          training_hooks=[training_utils.ETAHook(
              {} if config.use_tpu else dict(loss=model.loss),
              num_train_steps, config.iterations_per_loop, config.use_tpu, 10)])
    else:
      assert mode == tf.estimator.ModeKeys.PREDICT
      output_spec = tf.estimator.tpu.TPUEstimatorSpec(
          mode=mode,
          predictions=utils.flatten_dict(model.outputs),
          scaffold_fn=scaffold_fn)

    utils.log("Building complete")
    return output_spec

  return model_fn


class ModelRunner(object):
  """Fine-tunes a model on a supervised task."""

  def __init__(self, config: configure_finetuning.FinetuningConfig, tasks,
               pretraining_config=None):
    self._config = config
    self._tasks = tasks
    self._preprocessor = preprocessing.Preprocessor(config, self._tasks)

    is_per_host = tf.estimator.tpu.InputPipelineConfig.PER_HOST_V2
    tpu_cluster_resolver = None
    if config.use_tpu and config.tpu_name:
      tpu_cluster_resolver = tf.distribute.cluster_resolver.TPUClusterResolver(
          config.tpu_name, zone=config.tpu_zone, project=config.gcp_project)
    tpu_config = tf.estimator.tpu.TPUConfig(
        iterations_per_loop=config.iterations_per_loop,
        num_shards=config.num_tpu_cores,
        per_host_input_for_training=is_per_host,
        tpu_job_name=config.tpu_job_name)
    run_config = tf.estimator.tpu.RunConfig(
        cluster=tpu_cluster_resolver,
        model_dir=config.model_dir,
        save_checkpoints_steps=config.save_checkpoints_steps,
        save_checkpoints_secs=None,
        tpu_config=tpu_config)

    if self._config.do_train:
      (self._train_input_fn,self.train_steps) = self._preprocessor.prepare_train()
    else:
      self._train_input_fn, self.train_steps = None, 0
    model_fn = model_fn_builder(
        config=config,
        tasks=self._tasks,
        num_train_steps=self.train_steps,
        pretraining_config=pretraining_config)
    self._estimator = tf.estimator.tpu.TPUEstimator(
        use_tpu=config.use_tpu,
        model_fn=model_fn,
        config=run_config,
        train_batch_size=config.train_batch_size,
        eval_batch_size=config.eval_batch_size,
        predict_batch_size=config.predict_batch_size)

  def train(self):
    utils.log("Training for {:} steps".format(self.train_steps))
    self._estimator.train(
        input_fn=self._train_input_fn, max_steps=self.train_steps)

  def evaluate(self):
    return {task.name: self.evaluate_task(task) for task in self._tasks}

  def evaluate_task(self, task, split="dev", return_results=True):
    """Evaluate the current model."""
    utils.log("Evaluating", task.name)
    best_trial_info_file = os.path.join(self._config.model_dir, "best_trial.txt")
    eval_input_fn, _ = self._preprocessor.prepare_predict([task], split)
    def _best_trial_info():
        if tf.gfile.Exists(best_trial_info_file):
            with tf.gfile.GFile(best_trial_info_file, "r") as best_info:
                global_step, best_metric_global_step, metric_value = (best_info.read().split(":"))
                global_step = int(global_step)
                best_metric_global_step = int(best_metric_global_step)
                metric_value = float(metric_value)
        else:
            metric_value = -1
            best_metric_global_step = -1
            global_step = -1
        tf.logging.info("Best trial info: Step: %s, Best Value Step: %s, Best Value: %s", global_step, best_metric_global_step, metric_value)
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
    output_eval_file = os.path.join(self._config.model_dir,"eval_results.txt")
    key_name = "R10@1"
    global_step, best_perf_global_step, best_perf = _best_trial_info()
    writer = tf.gfile.GFile(output_eval_file, "w")
    steps_and_files = {}
    filenames = tf.gfile.ListDirectory(self._config.model_dir)
    for filename in filenames:
        if filename.endswith(".index"):
            ckpt_name = filename[:-6]
            cur_filename = os.path.join(self._config.model_dir, ckpt_name)
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
        result = self._estimator.predict(input_fn=eval_input_fn,checkpoint_path=checkpoint_path)
        score,label_list=[],[]
        for prediction in result:
            probabilities = prediction["conv_logits"].tolist()
            label=prediction["conv_label_ids"].tolist()
            score.append(probabilities[-1])
            label_list.append(label)
        if self._config.corpus=="mutual":
            result_metrics=MutualMetrics(score,label_list,count=4)
        else:
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
                tf.gfile.Rename(os.path.join(self._config.model_dir, src_ckpt),os.path.join(self._config.model_dir, tgt_ckpt),overwrite=True)
        else:
            _remove_checkpoint(checkpoint_path)
        writer.write("=" * 50 + "\n")
        writer.flush()
        with tf.gfile.GFile(best_trial_info_file, "w") as best_info:
            best_info.write("{}:{}:{}".format(global_step, best_perf_global_step, best_perf))
    writer.close()
    return  best_perf
    
    
        
#     eval_input_fn, _ = self._preprocessor.prepare_predict([task], split)
#     results = self._estimator.predict(input_fn=eval_input_fn,yield_single_examples=True)
#     scorer = task.get_scorer()
#     for r in results:
#       if r["task_id"] != len(self._tasks):  # ignore padding examples
#         r = utils.nest_dict(r, self._config.task_names)
#         scorer.update(r[task.name])
#     if return_results:
#       utils.log(task.name + ": " + scorer.results_str())
#       utils.log()
#       return dict(scorer.get_results())
#     else:
#       return scorer

  def write_classification_outputs(self, tasks, trial, split):
    """Write classification predictions to disk."""
    utils.log("Writing out predictions for", tasks, split)
    predict_input_fn, _ = self._preprocessor.prepare_predict(tasks, split)
    results = self._estimator.predict(input_fn=predict_input_fn,
                                      yield_single_examples=True)
    # task name -> eid -> model-logits
    logits = collections.defaultdict(dict)
    for r in results:
      if r["task_id"] != len(self._tasks):
        r = utils.nest_dict(r, self._config.task_names)
        task_name = self._config.task_names[r["task_id"]]
        logits[task_name][r[task_name]["eid"]] = (
            r[task_name]["logits"] if "logits" in r[task_name]
            else r[task_name]["predictions"])
    for task_name in logits:
      utils.log("Pickling predictions for {:} {:} examples ({:})".format(
          len(logits[task_name]), task_name, split))
      if trial <= self._config.n_writes_test:
        utils.write_pickle(logits[task_name], self._config.test_predictions(
            task_name, split, trial))


def write_results(config: configure_finetuning.FinetuningConfig, results):
  """Write evaluation metrics to disk."""
  utils.log("Writing results to", config.results_txt)
  utils.mkdir(config.results_txt.rsplit("/", 1)[0])
  utils.write_pickle(results, config.results_pkl)
  with tf.io.gfile.GFile(config.results_txt, "w") as f:
    results_str = ""
    for trial_results in results:
      for task_name, task_results in trial_results.items():
        if task_name == "time" or task_name == "global_step":
          continue
        results_str += task_name + ": " + " - ".join(
            ["{:}: {:.2f}".format(k, v)
             for k, v in task_results.items()]) + "\n"
    f.write(results_str)
  utils.write_pickle(results, config.results_pkl)


def run_finetuning(config: configure_finetuning.FinetuningConfig):
  """Run finetuning."""
  os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
  os.environ["CUDA_VISIBLE_DEVICES"] = str(config.gpu)
  # Setup for training
  results = []
  trial = 1
  heading_info = "model={:}, trial {:}/{:}".format(config.model_name, trial, config.num_trials)
  heading = lambda msg: utils.heading(msg + ": " + heading_info)
  heading("Config")
  utils.log_config(config)
  generic_model_dir = config.model_dir
  tasks = task_builder.get_tasks(config)
  # Train and evaluate num_trials models with different random seeds
  while config.num_trials < 0 or trial <= config.num_trials:
    config.model_dir = generic_model_dir + "_" + str(trial)
    if config.do_train:
      utils.rmkdir(config.model_dir)

    model_runner = ModelRunner(config, tasks)
    if config.do_train:
      heading("Start training")
      model_runner.train()
      utils.log()

    if config.do_eval:
      heading("Run dev set evaluation")
      model_runner.evaluate()
#       results.append(model_runner.evaluate())
#       write_results(config, results)
#       if config.write_test_outputs and trial <= config.n_writes_test:
#         heading("Running on the test set and writing the predictions")
#         for task in tasks:
#           # Currently only writing preds for GLUE and SQuAD 2.0 is supported
#           if task.name in ["cola", "mrpc", "mnli", "sst", "rte", "qnli", "qqp","sts","conv"]:
#             for split in task.get_test_splits():
#               model_runner.write_classification_outputs([task], trial, split)
#           elif task.name == "squad":
#             scorer = model_runner.evaluate_task(task, "test", False)
#             scorer.write_predictions()
#             preds = utils.load_json(config.qa_preds_file("squad"))
#             null_odds = utils.load_json(config.qa_na_file("squad"))
#             for q, _ in preds.items():
#               if null_odds[q] > config.qa_na_threshold:
#                 preds[q] = ""
#             utils.write_json(preds, config.test_predictions(
#                 task.name, "test", trial))
#           else:
#             utils.log("Skipping task", task.name,
#                       "- writing predictions is not supported for this task")

    if trial != config.num_trials and (not config.keep_all_models):
      utils.rmrf(config.model_dir)
    trial += 1


def main():
  parser = argparse.ArgumentParser(description=__doc__)
  parser.add_argument("--data_dir", required=True,help="Location of data files (model weights, etc).")
  parser.add_argument("--model_name", required=True,help="The name of the model being fine-tuned.")
  parser.add_argument("--corpus", required=True,help="The corpus of model")
  parser.add_argument("--model_path", required=True,help="the pretrained model path")
  parser.add_argument("--output_path", required=True,help="the fine-tune model")
  parser.add_argument("--tfrecords_dir",required=True,help="tfrecords file path")
  parser.add_argument("--gpu",required=True,help="gpu devices")
  parser.add_argument("--task_names",required=True,help="task name to train")
  parser.add_argument("--learning_rate",type=float,default=0.0001,help="the learning_rate")
  parser.add_argument("--num_train_epochs",type=int,default=3,help="the epochs to train")
  parser.add_argument("--save_checkpoints_steps",type=int,default=1000,help="save the model")
  parser.add_argument("--iterations_per_loop",type=int,default=1000,help="iterations_per_loop")
  parser.add_argument("--train_batch_size",type=int,default=2,help="train_batch_size")
  parser.add_argument("--eval_batch_size",type=int,default=2,help="eval_batch_size")
  parser.add_argument("--predict_batch_size",type=int,default=2,help="predict_batch_size")
  parser.add_argument("--max_seq_length",type=int,default=512,help="the max length for sequence")
  parser.add_argument('--do_train', action='store_true',help='train the model')
  parser.add_argument('--do_eval', action='store_true',help='eval the model')
  parser.add_argument('--do_predict', action='store_true',help='eval the model')
  parser.add_argument("--hparams", default="{}",help="JSON dict of model hyperparameters.")
  
  args = parser.parse_args()
  if args.hparams.endswith(".json"):
    hparams = utils.load_json(args.hparams)
  else:
    hparams = json.loads(args.hparams)
  tf.logging.set_verbosity(tf.logging.ERROR)
  #model_name,model_path,corpus,data_dir,tfrecords_dir,output_dir
  run_finetuning(configure_finetuning.FinetuningConfig(args=args))


if __name__ == "__main__":
  main()
