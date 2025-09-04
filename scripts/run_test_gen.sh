export HF_HOME="../hf_cache/"

export CUDA_VISIBLE_DEVICES=7

python src/test_gen.py \
    --checkpoint_path "./checkpoints/generative"

# python src/test_gen.py \
#     --checkpoint_path "./checkpoints/generative_annotation"