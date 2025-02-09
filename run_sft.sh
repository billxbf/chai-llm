pip install --upgrade transformers trl bitsandbytes accelerate peft scikit-learn deepspeed wandb

python login.py

accelerate launch --config-file "/config/deepspeed_z3_p4.yaml" trainer/sft.py \
    --tokenizer_path=/group-volume/binfeng/misc/chai/ckpt/nemo13b \
    --model_path=/group-volume/binfeng/misc/chai/ckpt/nemo13b \
    --dataset_path=/group-volume/binfeng/misc/chai/datasets/sft50k \
    --model_save_path=/group-volume/binfeng/misc/chai/ckpt/nemo13b-sft \
    --train_split=train \
    --epoch=1 \
    --lr=1e-5 \
    --bs=64 \
    --wd=1e-4 \
    --bs_per_device=8 \
    --save_only_model=True \
    --seed=42