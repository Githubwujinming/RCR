import logging
import os
import json
import argparse
import torch
import dataloaders
import models
import math
from utils import Logger, setup_logger
from utils.wandb_logger import WandbLogger
from trainer import Trainer
import torch.nn.functional as F
from utils.losses import abCE_loss, CE_loss, consistency_weight, FocalLoss, softmax_helper, get_alpha


def get_instance(module, name, config, *args):
    # GET THE CORRESPONDING CLASS / FCT 
    return getattr(module, config[name]['type'])(*args, **config[name]['args'])

def main(config, resume):
    torch.manual_seed(42)
    
    if config['enable_wandb']:
        import wandb
        print("Initializing wandblog.")
        wandb_logger = WandbLogger(config)
        # Training log
        wandb.define_metric('epoch')
        wandb.define_metric('training/train_step')
        wandb.define_metric("training/*", step_metric="train_step")
        # Validation log
        wandb.define_metric('validation/val_step')
        wandb.define_metric("validation/*", step_metric="val_step")
        # Initialization
        train_step = 0
        val_step = 0
    else:
        wandb_logger = None

    
    # DATA LOADERS
    config['train_supervised']['percnt_lbl'] = config["sup_percent"]
    config['train_unsupervised']['percnt_lbl'] = config["unsup_percent"]
    config['train_unsupervised']['use_weak_lables'] = config['use_weak_lables']
    supervised_loader = dataloaders.CDDataset(config['train_supervised'])
    unsupervised_loader = dataloaders.CDDataset(config['train_unsupervised'])
    val_loader = dataloaders.CDDataset(config['val_loader'])
    iter_per_epoch = len(unsupervised_loader)

    # SUPERVISED LOSS
    if config['model']['sup_loss'] == 'CE':
        sup_loss = CE_loss
    elif config['model']['sup_loss'] == 'FL':
        alpha = get_alpha(supervised_loader) # calculare class occurences
        print(alpha)
        sup_loss = FocalLoss(apply_nonlin = softmax_helper, alpha = alpha, gamma = 2.0, smooth = 1e-5)
    else:
        sup_loss = abCE_loss(iters_per_epoch=iter_per_epoch, epochs=config['trainer']['epochs'],
                                num_classes=val_loader.dataset.num_classes)

    # MODEL
    rampup_ends = int(config['ramp_up'] * config['trainer']['epochs'])
    cons_w_unsup = consistency_weight(final_w=config['unsupervised_w'], iters_per_epoch=len(unsupervised_loader),
                                        rampup_ends=rampup_ends)

    model = models.Consistency_ResNet50_CD(num_classes=val_loader.dataset.num_classes, conf=config['model'],
    						sup_loss=sup_loss, cons_w_unsup=cons_w_unsup,
    						weakly_loss_w=config['weakly_loss_w'], use_weak_lables=config['use_weak_lables'])
    print(f'\n{model}\n')

    # TRAINING
    trainer = Trainer(
        model=model,
        resume=resume,
        config=config,
        supervised_loader=supervised_loader,
        unsupervised_loader=unsupervised_loader,
        val_loader=val_loader,
        iter_per_epoch=iter_per_epoch,
        # train_logger=train_logger,
        wandb_logger=wandb_logger)

    trainer.train()

if __name__=='__main__':
    # PARSE THE ARGS
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_LEVIR_50.json',type=str,
                        help='Path to the config file')
    parser.add_argument('-r', '--resume', default=None, type=str,
                        help='Path to the .pth model checkpoint to resume training')
    parser.add_argument('-d', '--device', default=None, type=str,
                           help='indices of GPUs to enable (default: all)')
    parser.add_argument('--local', action='store_true', default=False)
    parser.add_argument('-enable_wandb', action='store_true', default=False)
    args = parser.parse_args()

    config = json.load(open(args.config))
    config['enable_wandb'] = args.enable_wandb
    torch.backends.cudnn.benchmark = True
    resume = config["resume_path"]
    main(config, resume)
