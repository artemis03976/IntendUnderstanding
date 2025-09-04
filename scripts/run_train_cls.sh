export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=7

python src/train_cls.py \
    --lr 1e-5 \
    --n_epochs 20 \
    --batch_size 32 \
    --log_path "./log/" \
    --output_path "./checkpoints"