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
       
       
       
       
       

