import os
import argparse
import json
from argparse import Namespace


def get_args():
    parser = argparse.ArgumentParser()

    # model settings
    parser.add_argument(
        '--model_name', type=str, default='distilbert-base-multilingual-cased', 
        help='base model'
    )

    # data settings
    parser.add_argument(
        '--n_labels', type=int, default=11, 
        help=''
    )
    parser.add_argument(
        '--data_path', type=str, default='./data',
        help='dataset path'
    )

    # training settings
    parser.add_argument(
        '--seed', type=int, default=42, 
        help='random seed'
    )
    parser.add_argument(
        '--lr', type=float, default=0.001,
        help='Learning rate of network.'
    )
    parser.add_argument(
        '--n_epochs', type=int, default=10, 
        help='Number of training epochs.'
    )
    parser.add_argument(
        '--batch_size', type=int, default=4,
        help='Training batch size. Set to n_samples by default.'
    )
    parser.add_argument(
        '--gradient_accumulation_steps', type=int, default=1,
        help='gradient accumulation steps during training'
    )

    # save settings
    parser.add_argument(
        '--log_path', type=str, default=None,
        help='path to log th results'
    )
    parser.add_argument(
        '--output_path', type=str, default='./checkpoints',
        help='path to save the model'
    )

    args = parser.parse_args()

    os.makedirs(args.output_path, exist_ok=True)
    # print and save the args
    with open(os.path.join(args.output_path, 'args.json'), 'w') as f:
        json.dump(vars(args), f, indent=4)

    return args


def get_test_args():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        '--checkpoint_path', type=str, default='./checkpoints',
        help='path to save the model'
    )

    args = parser.parse_args()

    with open(os.path.join(args.checkpoint_path, "args.json"), "r") as f:
        args_dict = json.load(f)
    base_args = Namespace(**args_dict)

    merged = vars(base_args).copy()
    merged.update(vars(args))

    return Namespace(**merged)