export DATA_DIR=dataset
export RECORD=recordset
export OUTPUT_DIR=OUTBERT
##############################
export CORPUS=douban  #ubuntu,douban,ecomm
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
       
       
       
       
       

