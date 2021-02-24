#coding=utf-8
'''
Created on 2020年7月22日

@author: Administrator
'''
import logging
import os
import sys
import copy
import jieba
import jieba.posseg as pseg
from tqdm import tqdm
from io import open
import json
from collections import OrderedDict
import glob
import random
import numpy as np
import re

logger = logging.getLogger(__name__)
pos_sets=["dg","d","Ag","a","n","ad","an","Ng","nr","ns","nt","nz","vg","v","vd","vn","q","m"]
from dictionary import BaiduChineseWordDictionary
bd_dic = BaiduChineseWordDictionary()
def get_syn_ant(token):
    result=bd_dic.query(token)
    antonyms=result["antonyms"]
    synonyms=result["synonyms"]
    return (antonyms,synonyms)

"""
construct the adversarial instance by replace the token in correct response with the context
tokens with the same POS
"""
def group_adversarial_instance(group_example,sim_type):
    candiates_sentences=[]
    contexts=" ".join(group_example[0].split("\t")[1:-1])
    words =pseg.cut(contexts)
    tokens_pos=[(w.word,w.flag) for w in words if w.word!=' ']

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
    golden_response=""
    for example_index,example in enumerate(group_example):
        instance=example.split("\t")
        label_id=instance[0]
        if int(label_id)==1:
            golden_response=instance[-1]
            ex_index=example_index
    if len(golden_response)<=0: 
        return [],0
#     response_list = golden_response.split()
    response_pos=pseg.cut(golden_response)
    response_pos=[(w.word,w.flag) for w in response_pos if w.word!=' ']
    response_list=[item[0] for item in response_pos]
    clear_reponse_index=OrderedDict()
    for _index,token in enumerate(response_pos):
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
                wordnet_synsets=get_syn_ant(source_token)[1].split()
            else:
                wordnet_synsets=get_syn_ant(source_token)[0].split()
            wordnet_synsets=set(wordnet_synsets)
            for pos,candidates_tokens in target_candiates.items():
                if len(candidates_tokens)>0:
                    candiates_result=wordnet_synsets.intersection(candidates_tokens)
                    if len(candiates_result)>0:
                        candiates_result=set(candiates_result)
                        for token in candiates_result:
                            temp_list=copy.deepcopy(response_list)
                            temp_list[_index]=token
                            words =pseg.cut(" ".join(temp_list))
                            tokens_pos=[(w.word,w.flag) for w in words if w.word!=' ']
                            after_pos=tokens_pos[_index][1]
                            if after_pos == pos:
                                candiates_sentences.append(" ".join(temp_list))
    return  candiates_sentences,ex_index
    
def adversarial_instance_by_context(inputFile,outputFile,similarity,sim_type="syn",corpus="mutual",count=4):
    dev_examples=[]
    outputFile=outputFile+sim_type+".txt"
    wf=open(outputFile,"w",encoding="utf-8")
    simFileName=os.path.join(similarity,"%s_adv_%s.txt"%(corpus,sim_type))
    wo=open(simFileName,"w",encoding="utf-8")
    with open(inputFile,"r",encoding="utf-8") as f:
        for line in f:
            line=line.rstrip()
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
                            wo.write(items[-1]+"\t"+candiate_example+"\n")
                            items[-1]=candiate_example
                            wf.write("\t".join(items)+"\n")
                    else:
                        items=example.split("\t")
                        if replace_id==example_index:
                            items[-1]=candiate_example
                        wf.write("\t".join(items)+"\n")
                 
    print("adv example %d"%(total))
        
if __name__=="__main__":
    adversarial_instance_by_context(inputFile="origin_dataset/ecomm/test.txt",
                                    outputFile="adversarial/ecomm/test_adv_",
                                    similarity="adversarial/ecomm",
                                    sim_type="syn",#syn,ant
                                    corpus="ecomm",
                                    count=10)
