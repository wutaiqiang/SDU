log_path=""
model="" # such as "allenai/scibert_scivocab_cased", please refer to the huggingface repo for aviable model
output_folder="" # folder to save the generate results


nohup python3 code/train.py \
    --input_path data/english/scientific \
    --output_path $output_folder\
    --train_batch_size 2 \
    --eval_batch_size 20 \
    --print_interval 100 \
    --eval_interval 1000 \
    --max_seq_len 256 \
    --bert_model $model \
    --learning_rate 1e-6 \
    --language en \
    --num_train_epochs 5 >$log_path 2>&1

echo "Done"