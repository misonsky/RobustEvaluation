### Evaluating the Robustness of Retrieval Based Dialogue Systems  
This repository  contains all the code and datasets  
### Download Dataset  
|  Corpus   | Link |
|  ----  | ----  |
| Mutual  | Link |
| Douban  | Link |
| E-comm  | Link |
| Ubuntu  | Link |
Description:  Each dataset folder contains four different adversarial datasets.  
### Dataset Statistics
|  Corpus   | original | stability |  sensitivity | diversity | extensibility |
|  ----  | ----  | ----  | ----  | ----  | ----  |
| Mutual  | 7,688 |  10,632 |  10,620 |  17,700 |  7,688 |
| Ubuntu  | 100,000 |  1,500,000 |  1,400,370 |  2,484,550 |  100,000 |
| Douban  | 10,000 |  30,000 |  24,720 |  42,950 |  10,000 |
| E-comm  | 10,000 |  30,000 |  19,110 |  33,950 |  10,000 |
###  Prepare data
> please put the data under the dataset folder
### Embedding
> For SMN, DUA, DAM, IOI and MSN Models, use the globe to train the corresponding word vector, the dimension of the vector is 300. [Glove](https://github.com/stanfordnlp/GloVe)   

###  Scripts
>Run SMN on MuTual  

```
export Dataset="dataset"
python run.py \
--data_dir=$Dataset \
--gpu=0 \
--corpus=mutual \
--max_turn=10 \
--max_utterance_len=50 \
--model=smn \
--batch_size=128 \
--learning_rate=0.001
```
> Run DUA on MuTual  

```
export Dataset="dataset"
python run.py \
--data_dir=$Dataset \
--gpu=0 \
--corpus=mutual \
--max_turn=10 \
--max_utterance_len=50 \
--model=dua \
--batch_size=128 \
--learning_rate=0.001
```
>  Run DAM on MuTual   

```
export Dataset="dataset"
python run.py \
--data_dir=$Dataset \
--gpu=0 \
--corpus=mutual \
--max_turn=10 \
--max_utterance_len=50 \
--model=dam \
--batch_size=128 \
--learning_rate=0.001
```
> Run IOI on MuTual   

```
export Dataset="dataset"
python run.py \
--data_dir=$Dataset \
--gpu=0 \
--corpus=mutual \
--max_turn=10 \
--max_utterance_len=50 \
--model=ioi \
--batch_size=128 \
--learning_rate=0.0001
```
> Run MSN on MuTual   
```
export Dataset="dataset"
python run.py \
--data_dir=$Dataset \
--gpu=0 \
--corpus=mutual \
--max_turn=10 \
--max_utterance_len=50 \
--model=msn \
--batch_size=128 \
--learning_rate=0.001
```

> Run BERT on Douban   
```
export DATA_DIR=dataset
export RECORD=recordset
export OUTPUT_DIR=OUTBERT
##############################
export CORPUS=douban  #ubuntu,douban,ecomm
export MODEL=BERT     #BERT
export BERTPATH=pretrainBERT/bert_base_ch
##############################
python BERTModel/BERT/run_conversation.py \
       --data_dir=$DATA_DIR \
       --gpu=0 \
       --bert_config_file=$BERTPATH/bert_config.json \
       --vocab_file=$BERTPATH/vocab.txt \
       --init_checkpoint=$BERTPATH/bert_model.ckpt \
       --task_name=conv \
       --record_dir=$RECORD \
       --corpus=$CORPUS \
       --model=$MODEL \
       --output_dir=$OUTPUT_DIR \
       --do_lower_case=True \
       --max_seq_length=52 \
       --do_prepare=False \
       --do_train=False \
       --do_eval=True \
       --do_predict=False \
       --train_batch_size=2 \
       --learning_rate=2e-5 \
       --save_checkpoints_steps=1000 \
       --iterations_per_loop=1000 \
       --num_train_epochs=10.0
```
> Run XLNET on Douban   
```
export DATA_DIR=dataset
export RECORD=recordset
export OUTPUT_DIR=OUTBERT
##############################
export CORPUS=douban
export MODEL=XLNET     #BERT
export BERTPATH=pretrainBERT/xlnet_base_ch
##############################
python BERTModel/XLNET/run_conversation.py \
       --data_dir=$DATA_DIR \
       --gpu=0 \
       --model_config_path=$BERTPATH/xlnet_config.json \
       --init_checkpoint=$BERTPATH/xlnet_model.ckpt \
       --spiece_model_file=$BERTPATH/spiece.model \
       --task_name=conv \
       --record_dir=$RECORD \
       --corpus=$CORPUS \
       --model=$MODEL \
       --output_dir=$OUTPUT_DIR \
       --do_lower_case=False \
       --max_seq_length=52 \
       --do_prepare=False \
       --do_train=False \
       --do_eval=True \
       --do_predict=False \
       --train_batch_size=2 \
       --learning_rate=5e-5 \
       --iterations=1000 \
       --save_checkpoints_steps=1000 \
       --num_train_epochs=10.0
```
> Run ALBERT on Douban  
```
export DATA_DIR=dataset
export RECORD=recordset
export OUTPUT_DIR=OUTBERT
##############################
export CORPUS=douban  #ubuntu,douban,ecomm
export MODEL=ALBERT     #ALBERT
export BERTPATH=pretrainBERT/albert_base_zh
##############################
python BERTModel/ALBERT/run_conversation.py \
       --data_dir=$DATA_DIR \
       --gpu=0 \
       --albert_config_file=$BERTPATH/albert_config.json \
       --vocab_file=$BERTPATH/vocab_chinese.txt \
       --init_checkpoint=$BERTPATH/model.ckpt-best \
       --task_name=conv \
       --record_dir=$RECORD \
       --corpus=$CORPUS \
       --model=$MODEL \
       --use_spm=False \
       --spm_model_file="" \
       --output_dir=$OUTPUT_DIR \
       --do_lower_case=True \
       --max_seq_length=52 \
       --do_prepare=False \
       --do_train=True \
       --do_eval=False \
       --do_predict=False \
       --train_batch_size=2 \
       --learning_rate=2e-5 \
       --save_checkpoints_steps=1000 \
       --iterations_per_loop=1000 \
       --num_train_epochs=2.0
```

> Run ELECTRA on Douban  
```
export DATA_DIR=dataset
export RECORD=recordset
export OUTPUT_DIR=OUTBERT
##############################
export CORPUS=mutual  #ubuntu,douban,ecomm,mutual
export MODEL=electra     #electra
export BERTPATH=pretrainBERT/electra_base_en
##############################
python BERTModel/electra-master/run_finetuning.py \
       --data_dir=$DATA_DIR \
       --model_name=$MODEL \
       --model_path=$BERTPATH \
       --output_path=$OUTPUT_DIR \
       --corpus=$CORPUS \
       --tfrecords_dir=$RECORD \
       --do_train \
       --do_eval \
       --gpu=0 \
       --task_names="conv" \
       --learning_rate=0.0001 \
       --num_train_epochs=3 \
       --save_checkpoints_steps=10 \
       --iterations_per_loop=10 \
       --train_batch_size=2 \
       --eval_batch_size=2 \
       --predict_batch_size=2 \
       --max_seq_length=60

```

> Run RoBerta on Douban
```
export DATA_DIR=dataset
export RECORD=recordset
export OUTPUT_DIR=OUTBERT
##############################
export CORPUS=douban  #douban,ecomm
export MODEL=roberta     #BERT
export BERTPATH=pretrainBERT/roberta_base_ch
##############################
python BERTModel/BERT/run_conversation.py \
       --data_dir=$DATA_DIR \
       --gpu=0 \
       --bert_config_file=$BERTPATH/bert_config.json \
       --vocab_file=$BERTPATH/vocab.txt \
       --init_checkpoint=$BERTPATH/bert_model.ckpt \
       --task_name=conv \
       --record_dir=$RECORD \
       --corpus=$CORPUS \
       --model=$MODEL \
       --output_dir=$OUTPUT_DIR \
       --do_lower_case=True \
       --max_seq_length=52 \
       --do_prepare=False \
       --do_train=False \
       --do_eval=True \
       --do_predict=False \
       --train_batch_size=2 \
       --learning_rate=2e-5 \
       --save_checkpoints_steps=1000 \
       --iterations_per_loop=1000 \
       --num_train_epochs=10.0
```
### Note
> IF run ALBERT Englilsh model please set "use_spm=true" and formulating "spm_model_file" parameters path.   
> IF run RoBerta English model please use the [transfomer](https://github.com/huggingface/transformers) library. 
