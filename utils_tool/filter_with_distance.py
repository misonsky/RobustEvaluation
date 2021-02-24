#coding=utf-8
'''

Created on Dec 11, 2020

@author: lyk

'''
from collections import OrderedDict
from utils.bleu import Bleu
import json
bleu_obj=Bleu(n=4)
def is_Chinese(word):
    for ch in word:
        if '\u4e00' <= ch <= '\u9fff':
            return True
    return False
def normalize(s):
    """
    Normalize strings to space joined chars.
    Args:
        s: a list of strings.
    Returns:
        A list of normalized strings.
    """
    if not s:
        return s
    normalized = []
    for ss in s:
        tokens = [c for c in list(ss) if len(c.strip()) != 0]
        normalized.append(' '.join(tokens))
    return normalized
def loadFile(FileName):
    with open(FileName,"r",encoding="utf-8") as f:
        datasets=json.load(f)
    for sample in datasets:
        yield sample
def compute_blue(FileName,outputFileName):
    orign_dict,paraph,scores={},{},{}
    final_sample=[]
    for sample in loadFile(FileName):
        temp_instance={}
        line_number=sample["line_number"]
        orign_qu=sample["Original Question"]
        paraph_ques=sample["Paraphrased Questions"]
        orign_dict[line_number]=orign_qu
        dist_bleu=1
        dist_para=None
        temp_instance["line_number"]=line_number
        temp_instance["Original Question"]=orign_qu
        temp_instance["Paraphrased Questions"]=[]
        for paraph_q in paraph_ques:
            if is_Chinese(paraph_q):
                paraph[line_number]=[normalize(paraph_q)]
                orign_dict[line_number]=[normalize(orign_qu)]
            else:
                paraph[line_number]=[" ".join([token.lower() for token in paraph_q])]
                orign_dict[line_number]=[" ".join([token.lower() for token in orign_qu])]
            bleu_scores,_=bleu_obj.compute_score(orign_dict, paraph)
            for i, bleu_score in enumerate(bleu_scores):
                scores['Bleu-%d' % (i + 1)] = bleu_score
            if scores["Bleu-4"]>0.8 and scores["Bleu-4"] < 0.9:
                if paraph_q not in temp_instance["Paraphrased Questions"]:
                    temp_instance["Paraphrased Questions"].append(paraph_q.lower())
        final_sample.append(temp_instance)
    with open(outputFileName,"w",encoding="utf-8") as f:
        json.dump(final_sample,f,ensure_ascii=False,indent=4)
if __name__=="__main__":
    compute_blue(FileName="paraphase/mutual/mutual_syn_response.json",outputFileName="paraphase/mutual/mutual_select_syn_response.json")
    
    
    





