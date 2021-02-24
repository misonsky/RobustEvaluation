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
