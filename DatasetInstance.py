#coding=utf-8
import os
import pickle as pkl
import numpy as np
from tqdm import tqdm
from collections import OrderedDict
import unicodedata
import tensorflow as tf
from tensorflow import keras

class DatasetInstance(object):
    def __init__(self,config,fileName):
        self.DataDict=OrderedDict()
        self.UNK="unk"
        self.config=config
        self.DataDict["text2id"]=self.construct_tokenize(fileName)
    def construct_tokenize(self,fileName):
        _,histories,response=self.create_dataset(fileName)
        text_list=[]
        for history in histories:
            text_list.extend(history)
        for res in list(response):
            text_list.extend(res)
        return self.tokenize(text_list)
    def unicode_to_ascii(self,s):
        return ''.join(c for c in unicodedata.normalize('NFD', s)
            if unicodedata.category(c) != 'Mn')
    def preprocess_sentence(self,w):
        w = self.unicode_to_ascii(w.lower().strip())
        return w
    def create_dataset(self,path):
        word_pairs=[]
        with open(path,'r',encoding="utf-8") as f:
            for line in tqdm(f):
                line=self.preprocess_sentence(line)
                line=line.rstrip()
                contents=line.split("\t")
                label=contents[0]
                history=contents[1:-1]
                response=contents[-1:]
                word_pairs.append([label,history,response])
        return zip(*word_pairs)
    def tokenize(self,text):
        """
        text can be a list/tuple of string:["this is a demo","this a demo"]
        or 2-D list words:[["this","is","a","demo"],["this","is","a","demo"]]
        """
        Tokenizer = tf.keras.preprocessing.text.Tokenizer(filters='',lower=True,oov_token=self.UNK)
        Tokenizer.fit_on_texts(text)
        return Tokenizer
    def sequence2id(self,sequence):
        return self.DataDict["text2id"].texts_to_sequences(sequence)
    def sequences2ids(self,sequences):
        results=list()
        for seq in sequences:
            results.append(self.sequence2id(seq))
        return results
    def id2sequence(self,sequence):
        return self.DataDict["text2id"].sequences_to_texts(sequence)
    def ids2sequences(self,sequences):
        results=list()
        for seq in sequences:
            results.append(self.id2sequence(seq))
        return results
    def pad_utterance(self,utterance):
        """
        utterance:2-D list
        """
        tensor=keras.preprocessing.sequence.pad_sequences(utterance,maxlen=self.config.max_utterance_len,padding='post',truncating='post')
        return tensor
    def pad_single_instance(self,history):
        num_turn=len(history)
        if num_turn < self.config.max_turn:#self.config.max_turn
            history=history + [[0]] *(self.config.max_turn-num_turn)
        elif num_turn > self.config.max_turn:
            history=history[num_turn-self.config.max_turn:]
        return history
    def pad_turn(self,histories):
        histories=[self.pad_single_instance(history) for history in histories]
        histories=[self.pad_utterance(history) for history in histories]
        return histories
    def TFRecoderFeature(self,path_to_file,outfile):
        writer=tf.data.experimental.TFRecordWriter(outfile)
        def create_int_feature(values):
            feature = tf.train.Feature(int64_list=tf.train.Int64List(value=list(values)))
            return feature 
        def serialize_example(history,response,labels):
            history=history.tolist()
            response=response.tolist()
            name_feature = {
                'history': create_int_feature(sum(history,[])),
                'response': create_int_feature(response),
                'labels': create_int_feature([labels])}
            tf_example=tf.train.Example(features=tf.train.Features(feature=name_feature))
            return tf_example.SerializeToString()
        labels,histories,responses=self.create_dataset(path_to_file)
        labels=[int(element) for element in labels]
        history_ids=self.sequences2ids(histories) #s * turn * seq_len
        history_ids=self.pad_turn(history_ids) # s * max_turn * max_seq_len
        response_ids=self.sequences2ids(responses)# s  * 1 * seq_len
        response_ids=[response[0] for response in response_ids]
        response_ids=self.pad_utterance(response_ids)# s* seq_len
        string_feature=[]
        for feature in zip(history_ids,response_ids,labels):
            string_feature.append(serialize_example(*feature))
        serialized_features_dataset=tf.data.Dataset.from_tensor_slices(string_feature)
        writer.write(serialized_features_dataset)
        tf.print("save the TFRecord file {}".format(outfile))
    def generate_embedding(self):
        word_index=self.DataDict["text2id"].word_index
        vocab_size=max(word_index.values())+1
        embedding_matrix = np.random.random((vocab_size, self.config.emb_size))
        embedding_matrix[0]=np.zeros(self.config.emb_size)
        pre_train_emb=os.path.join(self.config.emb_path,self.config.corpus,"vectors.txt")
        with open(pre_train_emb,"r",encoding="utf-8") as f:
            for line in tqdm(f):
                split_element=line.split()
                if self.config.lower_case:
                    token=split_element[0].lower()
                else:
                    token=split_element[0]
                vector=split_element[1:]
                assert len(vector)== self.config.emb_size
                if token in word_index:
                    embedding_matrix[word_index[token]]=np.asarray([float(item) for item in vector],dtype=np.float32)
        emb_file=os.path.join(self.config.data_dir,self.config.corpus,self.config.emb_file)
        with open(emb_file,"wb") as f:
            pkl.dump(embedding_matrix,f)
        tf.print("save the emb file {}".format(emb_file))
    def batch_data(self,config,recordFile,is_training=False):
        feature_description = {
                'history': tf.io.FixedLenFeature([config.max_turn * config.max_utterance_len], tf.int64),
                'response': tf.io.FixedLenFeature([config.max_utterance_len], tf.int64),
                'labels': tf.io.FixedLenFeature([],tf.int64)}
        def _parse_function(example):
            example= tf.io.parse_single_example(example,feature_description)
            for name in list(example.keys()):
                t=example[name]
                if t.dtype==tf.int64:
                    t=tf.cast(t,tf.int32)
                example[name]=t
            return example
        d=tf.data.TFRecordDataset(recordFile)
        if is_training:
            d=d.repeat(config.epochs)
            d=d.shuffle(buffer_size=100)
        parse_data=d.map(_parse_function,num_parallel_calls=tf.data.experimental.AUTOTUNE)
        parse_data = parse_data.prefetch(tf.data.experimental.AUTOTUNE).batch(config.batch_size)
        return parse_data
if __name__=="__main__":
    demo_obj=DatasetInstance(None,"dataset/douban/demo.txt")
    demo_obj.generate_embedding()
#     demo_obj.TFRecoderFeature("dataset/douban/demo.txt","dataset/douban/demo.tfrecord")

    