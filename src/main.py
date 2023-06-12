import os
import sys
import yaml

sys.path.append("/mnt/sdb/home/lwt/tao/HGCPep_new")
# config = yaml.load(open("../src/config_base.yaml", "r"), Loader=yaml.FullLoader)
# class_num = config['class_num']
# os.environ["CUDA_VISIBLE_DEVICES"] = "1,2,3"  # str(config['card_id'])

import time
from datetime import datetime

from dhg.random import set_seed
from src.Train_class import Trainer
import warnings
import torch
# torch.cuda.set_device(3)
warnings.filterwarnings("ignore")

"""
HGCPep 主函数
"""

if __name__ == '__main__':
    config = yaml.load(open("../src/config_base.yaml", "r"), Loader=yaml.FullLoader)
    os.environ["CUDA_VISIBLE_DEVICES"] = str(config['card_id'])
    print(config['task_name'], '\t\t\t', datetime.now().strftime('%b %d  %H : %M'))
    set_seed(config['seed'])
    print(config['loss_type'])
    if config['loss_type'] == 'naive_loss':
        print(config['naive_loss'])
    print('config -> ', config)
    since = time.time()

    trainer = Trainer(config, "../src/config_base.yaml")
    trainer.train()

    print("\nTraining finished . . . ")
    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(time_elapsed // 60, time_elapsed % 60))
