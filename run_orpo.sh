pip install --upgrade transformers trl bitsandbytes accelerate peft scikit-learn deepspeed wandb

python login.py

accelerate launch --config-file "config/deepspeed_z3_p4.yaml" trainer/orpo.py \
    --tokenizer_path=/group-volume/binfeng/misc/chai/ckpt/nemo13b-sft/checkpoint-738 \
    --model_path=/group-volume/binfeng/misc/chai/ckpt/nemo13b-sft/checkpoint-738 \
    --dataset_path=/group-volume/binfeng/misc/chai/datasets/preference20k \
    --model_save_path=/group-volume/binfeng/misc/chai/ckpt/nemo13b-sft-orpo \
    --train_split=train \
    --val_split=test \
    --epoch=2 \
    --lr=1e-5 \
    --bs=32 \
    --wd=1e-4 \
    --bs_per_device=8 \
    --save_only_model=True \
    --seed=42