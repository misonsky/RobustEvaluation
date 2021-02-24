#coding=utf-8
'''
Created on 2020年7月22日

@author: Administrator
'''
import logging
import os
import sys
import copy
from tqdm import tqdm
from io import open
import json
from collections import OrderedDict
import glob
import random
from nltk.corpus import wordnet as wn
import numpy as np
import re
import nltk
nltk.download('averaged_perceptron_tagger')
nltk.download('punkt')
nltk.download('wordnet')
logger = logging.getLogger(__name__)
pos_sets=["JJ","JJR","JJS","NN","NNS","NNP","NNPS","RB","RBR","RBS","VB","VBD","VBP","VBZ"]

class InputExample(object):
    """A single training/test example for multiple choice"""

    def __init__(self, example_id, contexts, endings, label=None):
        """Constructs a InputExample.

        Args:
            example_id: Unique id for the example.
            contexts: list of str. The untokenized text of the first sequence (context of corresponding question).
            question: string. The untokenized text of the second sequence (question).
            endings: list of str. multiple choice's options. Its length must be equal to contexts' length.
            label: (Optional) string. The label of the example. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.example_id = example_id
        self.contexts = contexts
        self.endings = endings
        self.label = label

class DataProcessor(object):
    """Base class for data converters for multiple choice data sets."""

    def get_train_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the train set."""
        raise NotImplementedError()

    def get_dev_examples(self, data_dir):
        """Gets a collection of `Iimport nltk
from nltk.corpus import stopwords
from nltk.corpus import brownnputExample`s for the dev set."""
        raise NotImplementedError()

    def get_test_examples(self, data_dir):
        """Gets a collection of `InputExample`s for the test set."""
        raise NotImplementedError()

    def get_labels(self):
        """Gets the list of labels for this data set."""
        raise NotImplementedError()
class MuTualProcessor(DataProcessor):
    """Processor for the MuTual data set."""
    def get_train_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} train".format(data_dir))
        file = os.path.join(data_dir, 'train')
        file = self._read_txt(file)
        return self._create_examples(file, 'train')

    def get_dev_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} dev".format(data_dir))
        file = os.path.join(data_dir, 'dev')
        file = self._read_txt(file)
        return self._create_examples(file, 'dev')

    def get_test_examples(self, data_dir):
        """See base class."""
        logger.info("LOOKING AT {} test".format(data_dir))
        file = os.path.join(data_dir, 'test')
        file = self._read_txt(file)
        return self._create_examples(file, 'test')

    def get_labels(self):
        """See base class."""
        return ["0", "1", "2", "3"]
    def _read_txt(self, input_dir):
        lines = []
        files = glob.glob(input_dir + "/*txt")
        for file in tqdm(files, desc="read files"):
            with open(file, 'r', encoding='utf-8') as fin:
                data_raw = json.load(fin)
                lines.append(data_raw)
        return lines
    def _create_examples(self, lines, set_type):
        """Creates examples for the training and dev pos_setssets."""
        examples = []
        for (_, data_raw) in enumerate(lines):
            id = "%s-%s" % (set_type, data_raw["id"])
            article = data_raw["article"]
            truth = str(ord(data_raw['answers']) - ord('A'))
            options = data_raw['options']
            content=re.split("f :|m :",article)
            content=[item.rstrip() for item in content if len(item)>0]
            string_content=" ".join(content)
            examples.append(
                InputExample(
                    example_id=id,
                    contexts=string_content, # this is not english_punctuationsefficient but convenient
                    endings=[options[0], options[1], options[2], options[3]],
                    label=truth))
        return examples
def antsets_wordNet_tool(token):
    ant_set=set()
    syn=wn.synsets(token)
    for example in syn:
        for instance in example.lemmas():
            for ant_exam in instance.antonyms():
                ant_set.add(ant_exam.name())
    return ant_set
def synsets_wordNet_tool(token):
    syn_set=set()
    syn=wn.synsets(token)
    for example in syn: 
        for instance in example.lemmas():
            syn_token=instance.name()
            if syn_token!=token:
                syn_set.add(syn_token)
    return syn_set
    
        
    
"""
construct the adversarial instance by replace the token in correct response with the context
tokens with the same POS
"""
english_punctuations = [',', '.', ':', ';', '?', '(', ')', '[', ']', '&', '!', '*', '@', '#', '$', '%']
def group_adversarial_instance(group_example,sim_type):
    candiates_sentences=[]
    contexts=" ".join(group_example[0].split("\t")[1:-1])
    text_list = nltk.word_tokenize(contexts)
    text_list = [word for word in text_list if word not in english_punctuations]
    tokens_pos=nltk.pos_tag(text_list)
    context_dict=OrderedDict()
    for token_pos in tokens_pos:
        pos_tag=token_pos[1]
        token=token_pos[0]
        if pos_tag not in context_dict:
            context_dict[pos_tag]=set()
            context_dict[pos_tag].add(token)
        else:
            context_dict[pos_tag].add(token)
    ex_index = -1
    for example_index,example in enumerate(group_example):
        instance=example.split("\t")
        label_id=instance[0]
        if int(label_id)==1:
            golden_response=instance[-1]
            ex_index=example_index
    response_list = nltk.word_tokenize(golden_response)
    reponse_index_map=OrderedDict()
    for _index,token in enumerate(response_list):
        reponse_index_map[_index]=token
    response_pos=nltk.pos_tag(response_list)
    clear_reponse_index=OrderedDict()
    for _index,token in enumerate(response_pos):
        if token[0] not in  english_punctuations:
            clear_reponse_index[_index]=token
    tokens_candiates=OrderedDict()
    for _index,tuple_element in clear_reponse_index.items():
        token_pos=tuple_element[1]
        token=tuple_element[0]
        tokens_candiates[_index]=OrderedDict()
        if token not in tokens_candiates[_index]:
            tokens_candiates[_index][token]=OrderedDict()
        if token_pos not in tokens_candiates[_index][token]:
            tokens_candiates[_index][token][token_pos]=set()
        if token_pos in context_dict and token_pos in pos_sets:
            candiates_tokens=context_dict[token_pos]
            tokens_candiates[_index][token][token_pos]=candiates_tokens
    for _index,candiate_dict in tokens_candiates.items():
        for source_token,target_candiates in candiate_dict.items():
            if sim_type=="syn":
                wordnet_synsets=synsets_wordNet_tool(source_token)
            else:
                wordnet_synsets=antsets_wordNet_tool(source_token)
            for pos,candidates_tokens in target_candiates.items():
                if len(candidates_tokens)>0:
                    candiates_result=wordnet_synsets.intersection(candidates_tokens)
                    if len(candiates_result)>0:
                        candiates_result=set(candiates_result)
                        for token in candiates_result:
                            temp_list=copy.deepcopy(response_list)
                            temp_list[_index]=token
                            tokens_pos=nltk.pos_tag(temp_list)
                            after_pos=tokens_pos[_index][1]
                            if after_pos == pos:
                                candiates_sentences.append(" ".join(temp_list))
    return  candiates_sentences,ex_index
    
def adversarial_instance_by_context(inputFile,outputFile,similarity,sim_type="syn",corpus="mutual",count=4):
    dev_examples=[]
    outputFile=outputFile+sim_type+".txt"
    wf=open(outputFile,"w",encoding="utf-8")
#     simFileName=os.path.join(similarity,"%s_adv_%s.txt"%(corpus,sim_type))
#     wo=open(simFileName,"w",encoding="utf-8")
    with open(inputFile,"r",encoding="utf-8") as f:
        for line in f:
            dev_examples.append(line)
    total=0
    for i in tqdm(range(0,len(dev_examples),count)):
        group_dataset=dev_examples[i:i+count]
        candiates_sentences,ex_index=group_adversarial_instance(group_dataset,sim_type)
        if len(candiates_sentences) >0:
            for candiate_example in candiates_sentences:
                _index=[i for i in range(count)]
                _index.remove(ex_index)
                replace_id=random.sample(_index, 1)[0]
                for example_index,example in enumerate(group_dataset):
                    total+=1 
                    example=example.rstrip()
                    if sim_type=="syn":
                        if example_index!=ex_index:
                            wf.write(example+"\n")
                        else:
                            items=example.split("\t")
#                             wo.write(items[-1]+"\t"+candiate_example+"\n")
                            items[-1]=candiate_example
                            wf.write("\t".join(items)+"\n")
                    else:
                        items=example.split("\t")
                        if replace_id==example_index:
                            items[-1]=candiate_example
                        wf.write("\t".join(items)+"\n")
                 
    print("adv example %d"%(total))
        
if __name__=="__main__":
    adversarial_instance_by_context(inputFile="origin_dataset/ubuntu/test.txt",
                                    outputFile="adversarial/ubuntu/test_adv_",
                                    similarity="adversarial/ubuntu",
                                    sim_type="ant",#syn,ant
                                    corpus="ubuntu",
                                    count=10)
