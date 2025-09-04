export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=1

# python src/train_gen.py \
#     --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
#     --mode "classify" \
#     --data_path "./data/instructions.jsonl" \
#     --lr 2e-4 \
#     --n_epochs 3 \
#     --batch_size 4 \
#     --output_path "./checkpoints/generative"

python src/train_gen.py \
    --model_name "Qwen/Qwen2.5-1.5B-Instruct" \
    --mode "annotation" \
    --data_path "./data/annotation.jsonl" \
    --lr 2e-4 \
    --n_epochs 3 \
    --batch_size 4 \
    --output_path "./checkpoints/generative_annotation"