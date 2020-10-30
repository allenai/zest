#!/bin/bash

# ./bin/train_bart_run_eval.sh /path/to/zest/data learning_rate num_epochs

DATADIR=$1
shift;
LR=$1
shift;
EPOCHS=$1

conda activate zest_bart

OUTDIR=bart_large_${LR}_${EPOCHS}epochs_smoothing0.1

echo "TRAINING $LR $EPOCHS $OUTDIR"

python bin/fine_tune_bart.py \
    --task=zest \
    --data_dir=$DATADIR \
    --model_name_or_path=facebook/bart-large \
    --learning_rate=$LR \
    --train_batch_size=8 \
    --gradient_accumulation_steps=4 \
    --accumulate_grad_batches=4 \
    --eval_batch_size=1 \
    --num_train_epochs=$EPOCHS \
    --check_val_every_n_epoch=100 \
    --warmup_steps=100 \
    --gpus=1 \
    --do_train \
    --do_predict \
    --early_stopping_patience=-1 \
    --max_grad_norm=0.1 \
    --gradient_clip_val=0.1 \
    --fp16 \
    --fp16_opt_level=O2 \
    --weight_decay=0.01 \
    --adam_epsilon=1e-8 \
    --lr_scheduler=linear \
    --dropout=0.1 \
    --attention_dropout=0.1 \
    --max_source_length=512 \
    --max_target_length=64 \
    --val_max_target_length=64 \
    --test_max_target_length=64 \
    --eval_beams 4 \
    --eval_max_gen_length=64 \
    --row_log_interval=320 \
    --num_sanity_val_steps=0 \
    --n_val -1 \
    --freeze_embeds \
    --output_dir=$OUTDIR \
    --label_smoothing 0.1

# This evaluates on the dev set and writes the predictions to a file.
python bin/fine_tune_bart.py \
     --evaluate_only \
     --output_dir=$OUTDIR \
     --model_name_or_path ignore --data_dir ignore

# Run the official eval script.
python bin/evaluate-zest.py \
    --predictions-path $OUTDIR/val_preds.txt \
    --dev-path $DATADIR/dev.jsonl \
    --output-path $OUTDIR/val_preds_results_

