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
       
       
       
       
       

