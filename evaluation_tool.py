#coding=utf-8
from collections import OrderedDict
#MAP
def mean_average_precision(sort_data):
    #to do
    count_1 = 0
    sum_precision = 0
    for index in range(len(sort_data)):
        if sort_data[index][1] == 1:
            count_1 += 1
            sum_precision += 1.0 * count_1 / (index+1)
    return sum_precision / count_1
#MRR
def mean_reciprocal_rank(sort_data):
    sort_lable = [s_d[1] for s_d in sort_data]
    assert 1 in sort_lable
    return 1.0 / (1 + sort_lable.index(1))
#P@1
def precision_at_position_1(sort_data):
    if sort_data[0][1] == 1:
        return 1
    else:
        return 0
#R10@k
def recall_at_position_k_in_10(sort_data, k):
    sort_lable = [s_d[1] for s_d in sort_data]
    select_lable = sort_lable[:k]
    return 1.0 * select_lable.count(1) / sort_lable.count(1)
# douban evaluation metrics
def evaluation_one_session(data):
    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    m_a_p = mean_average_precision(sort_data)
    m_r_r = mean_reciprocal_rank(sort_data)
    p_1 = precision_at_position_1(sort_data)
    r_1 = recall_at_position_k_in_10(sort_data, 1)
    r_2 = recall_at_position_k_in_10(sort_data, 2)
    r_5 = recall_at_position_k_in_10(sort_data, 5)
    return m_a_p, m_r_r, p_1, r_1, r_2, r_5

def DoubanMetrics(scores,labels,count = 10):
    eval_metrix=OrderedDict()
    R1,R2,R5,MRR,MAP,P1= 0,0,0,0,0,0
    total = 0
    assert len(scores)==len(labels)
    for i in range(0,len(scores),count):
        data=[]
        g_score=scores[i:i+count]
        g_label=labels[i:i+count]
        for score,label in zip(g_score,g_label):
            data.append((score,label))
        if 1 in g_label:
            total = total+1
            _map,mrr,p1,r1, r2, r5=evaluation_one_session(data)
            MAP +=_map
            MRR +=mrr
            R1 +=r1
            R2 +=r2
            R5 +=r5
            P1 +=p1
    eval_metrix["R10@1"]=R1*1.0 / total
    eval_metrix["R10@2"]=R2*1.0 / total
    eval_metrix["R10@5"]=R5*1.0 / total
    eval_metrix["P@1"]=P1*1.0 / total
    eval_metrix["MRR"]=MRR*1.0 / total
    eval_metrix["MAP"]=MAP*1.0 / total
    return eval_metrix
def evaluation_mutual_session(data):
    sort_data = sorted(data, key=lambda x: x[0], reverse=True)
    m_a_p = mean_average_precision(sort_data)
    m_r_r = mean_reciprocal_rank(sort_data)
    p_1 = precision_at_position_1(sort_data)
    r_1 = recall_at_position_k_in_10(sort_data, 1)
    r_2 = recall_at_position_k_in_10(sort_data, 2)
    r_4 = recall_at_position_k_in_10(sort_data, 4)
    return m_a_p, m_r_r, p_1, r_1, r_2, r_4
def groupMetrics(scores,labels,count = 10):
    group_example=5
    total = 0
    g_1,g_2,g_3,g_4,g_5=0,0,0,0,0
    group_container=[]
    for i in range(0,len(scores),count):
        data=[]
        g_score=scores[i:i+count]
        g_label=labels[i:i+count]
        for score,label in zip(g_score,g_label):
            data.append((score,label))
        if 1 in g_label:
            _,_,_,r1, _, _=evaluation_one_session(data)
            group_container.append(int(r1))
        if len(group_container)==group_example:
            total = total+1
            if sum(group_container) >=5:
                g_5 +=1
            if sum(group_container) >=4:
                g_4 +=1
            if sum(group_container) >=3:
                g_3 +=1
            if sum(group_container) >=2:
                g_2 +=1
            if sum(group_container) >=1:
                g_1 +=1
            group_container=[]
    g_1 =g_1 *1.0 /total
    g_2 =g_2 *1.0 /total
    g_3 =g_3 *1.0 /total
    g_4 =g_4 *1.0 /total
    g_5 =g_5 *1.0 /total
    print("g_1 {} g_2 {} g_3 {} g_4 {} g_5 {}".format(g_1,g_2,g_3,g_4,g_5))
def MutualMetrics(scores,labels,count = 4):
    eval_metrix=OrderedDict()
    R1,R2,R5,MRR,MAP,P1= 0,0,0,0,0,0
    total = 0
    assert len(scores)==len(labels)
    for i in range(0,len(scores),count):
        data=[]
        g_score=scores[i:i+count]
        g_label=labels[i:i+count]
        for score,label in zip(g_score,g_label):
            data.append((score,label))
        if 1 in g_label:
            total = total+1
            _map,mrr,p1,r1, r2, r5=evaluation_mutual_session(data)
            MAP +=_map
            MRR +=mrr
            R1 +=r1
            R2 +=r2
            R5 +=r5
            P1 +=p1
    eval_metrix["R10@1"]=R1*1.0 / total
    eval_metrix["R10@2"]=R2*1.0 / total
    eval_metrix["R10@5"]=R5*1.0 / total
    eval_metrix["P@1"]=P1*1.0 / total
    eval_metrix["MRR"]=MRR*1.0 / total
    eval_metrix["MAP"]=MAP*1.0 / total
    return eval_metrix



