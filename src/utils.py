import random
import numpy as np
import torch
import torch.nn as nn
import os
from datetime import datetime
import logging
from torch.nn.parallel import DistributedDataParallel
import json
import re


def set_seeds(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)  # CPU random seed
    torch.cuda.manual_seed(seed)  # GPU random seed

    torch.backends.cudnn.enabled = True
    torch.backends.cudnn.benchmark = True


def get_logger(args):
    logger = logging.getLogger()
    logger.setLevel(logging.INFO)

    if logger.hasHandlers():
        logger.handlers.clear()

    formatter = logging.Formatter('%(asctime)s - %(levelname)s - %(message)s')

    if args.log_path is not None:
        os.makedirs(args.log_path, exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        file_path = os.path.join(args.log_path, f'train_log_{timestamp}.log')
        file_handler = logging.FileHandler(file_path, mode='w')
        file_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
    else:
        console_handler = logging.StreamHandler()
        console_handler.setFormatter(formatter)
        logger.addHandler(console_handler)

    return logger


def distribution_state_manager(model):
    if isinstance(model, DistributedDataParallel):
        return model.module
    else:
        return model


def clean_text(text):
    pattern = r'^\s*(?:```|~~~)\s*json\s*\n(.*?)\n\s*(?:```|~~~)\s*$'
    match = re.search(pattern, text, re.DOTALL | re.MULTILINE)
    
    if match:
        json_str =  match.group(1)
        json_str = re.sub(r'\s+', ' ', json_str.strip())
        json_str = json_str.replace('\\', '\\\\')
        return json.loads(json_str)