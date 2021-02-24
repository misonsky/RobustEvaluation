#coding=utf-8
import argparse
import os
import time
import pickle as pkl
from evaluation_tool import DoubanMetrics
from evaluation_tool import MutualMetrics
from evaluation_tool import groupMetrics
from DatasetInstance import DatasetInstance
from models.smn import SMN
from models.dam import DAM
from models.dua import DUA
from models.ioi import IOI
from models.msn import MSN
import tensorflow as tf
from tensorflow import keras
parser = argparse.ArgumentParser('robust evalation config for conversation')
parser.add_argument('--data_dir',type=str,default="dataset",help='dataset path')
parser.add_argument('--gpu',type=str,default="0",help='gpu devices')
parser.add_argument('--corpus',choices=["ubuntu","douban","ecomm","mutual"],default="douban",help='dataset')
parser.add_argument('--question',choices=["sta","sen","div","ext"],default="sta",help='question')
parser.add_argument('--emb_path',type=str,default="embeddings",help='embeddings path')
parser.add_argument('--emb_file',type=str,default="emb.pkl",help='embeddings file')
parser.add_argument('--lower_case',type=bool,default=True,help="lowe case for token")
parser.add_argument('--train_files', type=str,default='train.txt',help='train data')
parser.add_argument('--dev_files', type=str,default='test.txt',help='dev data')
parser.add_argument('--test_files', type=str,default='test.txt',help='text data')
parser.add_argument('--train_record', type=str,default='train.tfrecord',help='train recordfile')
parser.add_argument('--dev_record', type=str,default='dev.tfrecord',help='dev recordfile')
parser.add_argument('--test_record', type=str,default='test.tfrecord',help='test recordfile')
parser.add_argument('--prepare',action="store_true",help="train model")
parser.add_argument('--train',action="store_true",help="prepare dataset")
parser.add_argument('--eval',action="store_true",help="eval model")
dmn_settings = parser.add_argument_group('dmn model settings')
dmn_settings.add_argument("--filter_size",nargs='+',default=[32,16],help="the filter size of conv")
dmn_settings.add_argument("--num_layer",type=int,default=5,help="stack number of the layers")
ioi_settings = parser.add_argument_group('ioi model settings')
ioi_settings.add_argument("--ioi_number",type=int,default=7,help="the number of stack interaction layers")
msn_settings = parser.add_argument_group('msn model settings')
msn_settings.add_argument("--hop_num",type=int,default=3,help="the hop number")
msn_settings.add_argument("--msn_filter",nargs='+',default=[16,32,64],help="the filter size of conv")
parser.add_argument("--max_turn",type=int,default=10,help="max number turn of conversation")
parser.add_argument("--model",choices=['smn','dam','dua','ioi','msn'],default="ioi",help="the model name")
parser.add_argument("--max_utterance_len",type=int,default=50,help="max length of utterance")
parser.add_argument("--eval_step",type=int,default=5,help="numbert steps eval the model")
parser.add_argument("--log_step",type=int,default=5,help="numbert steps log info")
parser.add_argument('--batch_size',type=int,default=32,help="train batch size")
parser.add_argument('--learning_rate',type=float,default=0.001,help="learning")
parser.add_argument('--dropout',type=float,default=0.2,help="dropout rate")
parser.add_argument('--hidden_size',type=int,default=200,help="hidden size of lstm")
parser.add_argument('--emb_size',type=int,default=200,help="embedding size")
parser.add_argument('--epochs',type=int,default=10,help="train epochs")
parser.add_argument('--model_dir',type=str,default='TrainModel',help="path to save model")

config=parser.parse_args()
loss_object = keras.losses.SparseCategoricalCrossentropy(
    from_logits=True, reduction='none')
@tf.function
def loss_function(real, pred,clip_value=10):
    loss_ = loss_object(real, pred)
    return tf.reduce_mean(loss_)
def validation_parameters(config):
    if not config.train and not config.eval:
        raise ValueError("must specify one of them")
    model_dir=os.path.join(config.model_dir,config.corpus)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)

def learning_rate_schedule():
    if config.model !="msn":
        return config.learning_rate
    else:
        return keras.optimizers.schedules.ExponentialDecay(initial_learning_rate=config.learning_rate,
                                                     decay_steps=14000,
                                                     decay_rate=0.5)
def get_optimizer():
    optimizer = tf.keras.optimizers.Adam(learning_rate_schedule())
    return optimizer

def get_model(config,vocab_size):
    models={"smn":SMN,"dam":DAM,"dua":DUA,"ioi":IOI,"msn":MSN}
    emb_file=os.path.join(config.data_dir,config.corpus,config.emb_file)
    with open(emb_file,"rb") as f:
        embedding_matrix=pkl.load(f)
    train_model=models[config.model]
    return train_model(vocab_size=vocab_size,embedding_matrix=embedding_matrix,config=config)
@tf.function
def train_step(features,model,optimizer):
    his=features["history"]
    res=features["response"]
    labels=features["labels"]
    his=tf.reshape(his,shape=[res.shape[0],config.max_turn,config.max_utterance_len])
    with tf.GradientTape() as tape:
        logits,y_pre=model(his,res)
        if config.model=="ioi":
            loss_list=[]
            for logit in logits:
                loss_list.append(loss_function(labels, logit))
            batch_loss=sum(loss_list)
        else:
            batch_loss=loss_function(labels, logits)
    variables=model.trainable_variables
    gradients = tape.gradient(batch_loss, variables)
    optimizer.apply_gradients(zip(gradients, variables))
    if config.model=="ioi":
        y_pre=sum(y_pre)
    return batch_loss,y_pre
def prepare():
    train_file=os.path.join(config.data_dir,config.corpus,config.train_files)
    train_record=os.path.join(config.data_dir,config.corpus,config.train_record)
    dev_file=os.path.join(config.data_dir,config.corpus,config.dev_files)
    dev_record=os.path.join(config.data_dir,config.corpus,config.dev_record)
    test_file=os.path.join(config.data_dir,config.corpus,config.test_files)
    test_record=os.path.join(config.data_dir,config.corpus,config.test_record)
    if not os.path.isfile(os.path.join(config.data_dir,config.corpus,"dataInstance.pkl")):
        datasetInstance=DatasetInstance(config,train_file)
        with open(os.path.join(config.data_dir,config.corpus,"dataInstance.pkl"),"wb") as f:
            pkl.dump(datasetInstance,f)
    else:
        with open(os.path.join(config.data_dir,config.corpus,"dataInstance.pkl"),"rb") as f:
            datasetInstance=pkl.load(f)
    datasetInstance.generate_embedding()
    datasetInstance.TFRecoderFeature(train_file, train_record)
    datasetInstance.TFRecoderFeature(dev_file, dev_record)
    datasetInstance.TFRecoderFeature(test_file, test_record)
def evaluate(model,fileName,test_loss):
    if os.path.isfile(os.path.join(config.data_dir,config.corpus,"dataInstance.pkl")):
        with open(os.path.join(config.data_dir,config.corpus,"dataInstance.pkl"),"rb") as f:
            datasetInstance=pkl.load(f)
    else:
        tf.print("lack of datasetInstance object")
        return
    test_loss.reset_states()
    score,label_list=[],[]
    dev_dataset=datasetInstance.batch_data(
        config=config,
        recordFile=fileName,
        is_training=False)
    for features in dev_dataset:
        his=features["history"]
        res=features["response"]
        labels=features["labels"]
        his=tf.reshape(his,shape=[res.shape[0],config.max_turn,config.max_utterance_len])
        logits,y_pre=model(his,res)
        if config.model=="ioi":
            y_pre=sum(y_pre)
            loss_list=[]
            for logit in logits:
                loss_list.append(loss_function(labels, logit))
            batch_loss=sum(loss_list)
        else:
            batch_loss=loss_function(labels, logits)
        batch_score=y_pre[:,1].numpy().tolist()
        test_loss(batch_loss)
        score.extend(batch_score)
        label_list.extend(labels.numpy().tolist())
    if config.corpus=="mutual":
        eval_metrics=MutualMetrics(score,label_list,count=4)
    else:
        eval_metrics=DoubanMetrics(score,label_list,count=10)
    if config.question=="div":
        if config.corpus=="mutual":
            groupMetrics(score, label_list, count=4)
        else:
            groupMetrics(score, label_list, count=10)
    eval_metrics["eval_loss"]=test_loss.result()
    return eval_metrics
def train():
    train_record=os.path.join(config.data_dir,config.corpus,config.train_record)
    dev_record=os.path.join(config.data_dir,config.corpus,config.dev_record)
    model_dir=os.path.join(config.model_dir,config.corpus,config.model)
    with open(os.path.join(config.data_dir,config.corpus,"dataInstance.pkl"),"rb") as f:
        datasetInstance=pkl.load(f)
    tf.print("Info load data object...............")
    train_dataset=datasetInstance.batch_data(
        config=config,
        recordFile=train_record,
        is_training=True)
    tf.print("generate batch dataset...................")
    max_word_index=max(datasetInstance.DataDict["text2id"].word_index.values())
    optimizer=get_optimizer()
    model=get_model(config,vocab_size=max_word_index+1)
    checkpoint_prefix = os.path.join(model_dir, "ckpt")
    checkpoint = tf.train.Checkpoint(optimizer=optimizer,model=model)
    ckpt_manager = tf.train.CheckpointManager(checkpoint,checkpoint_prefix, max_to_keep=5)
    train_loss=tf.keras.metrics.Mean(name='train_loss')
    test_loss=tf.keras.metrics.Mean(name='test_loss')
    MAX_ACCURACY=-1
    train_loss.reset_states()
    for (batch, features) in enumerate(train_dataset,1):
        batch_loss,_=train_step(features,model,optimizer)
        train_loss(batch_loss)
        if batch % config.log_step==0:
            tf.print("log info {} to step {} train loss {}".format((batch-config.log_step),batch,train_loss.result()))
            train_loss.reset_states()
        if batch %config.eval_step==0:
            eval_metrics=evaluate(model, dev_record,test_loss)
            tf.print("step {} MAP {} MRR {} P@1 {} R10@1 {} R10@2 {} R10@5 {}".format(batch,eval_metrics["MAP"],eval_metrics["MRR"],eval_metrics["P@1"],eval_metrics["R10@1"],eval_metrics["R10@2"],eval_metrics["R10@5"]))
            if eval_metrics["R10@1"] >=MAX_ACCURACY:
                MAX_ACCURACY=eval_metrics["R10@1"]
                ckpt_save_path = ckpt_manager.save()
                tf.print("save the model {}".format(ckpt_save_path))
if __name__=="__main__":
    
    os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
    os.environ["CUDA_VISIBLE_DEVICES"] = config.gpu
    if config.train:
        train()
    elif config.prepare:
        prepare()
    elif config.eval:
        test_file=os.path.join(config.data_dir,config.corpus,config.question,config.test_files)
        test_record=os.path.join(config.data_dir,config.corpus,config.question,config.test_record)
        with open(os.path.join(config.data_dir,config.corpus,"dataInstance.pkl"),"rb") as f:
            datasetInstance=pkl.load(f)
        if not os.path.isfile(test_record):
            datasetInstance.TFRecoderFeature(test_file, test_record)
        max_word_index=max(datasetInstance.DataDict["text2id"].word_index.values())
        model=get_model(config,vocab_size=max_word_index+1)
        checkpoint = tf.train.Checkpoint(optimizer=get_optimizer(),
                                 model=model)
        model_dir=os.path.join(config.model_dir,config.corpus)
        checkpoint.restore(tf.train.latest_checkpoint(model_dir))
        test_loss=tf.keras.metrics.Mean(name='test_loss')
        test_accuracy = tf.keras.metrics.SparseCategoricalAccuracy(name='test_accuracy')
        eval_metrics=evaluate(model, test_record,test_loss)
        tf.print("MAP {} MRR {} P@1 {} R10@1 {} R10@2 {} R10@5 {}".format(eval_metrics["MAP"],eval_metrics["MRR"],eval_metrics["P@1"],eval_metrics["R10@1"],eval_metrics["R10@2"],eval_metrics["R10@5"]))
    
        
        
    
    
