import argparse
import logging
import scipy, math
from scipy import ndimage
import cv2
import numpy as np
import sys
import PIL
import json
import models
import dataloaders
from utils.helpers import colorize_mask, dir_exists
from utils.logger import setup_logger
from utils.pallete import get_voc_pallete
from utils import metrics
import torch
import torch.nn as nn
from torchvision import transforms
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset
import os
from tqdm import tqdm
from math import ceil
from PIL import Image
from pathlib import Path
from utils.metrics import eval_metrics, AverageMeter
from utils.htmlwriter import HTML
from matplotlib import pyplot as plt
from utils.helpers import DeNormalize
import time

def get_imgid_list(Dataset_Path, split, i):
    file_list  = os.path.join(Dataset_Path, 'list', split +".txt")
    filelist   = np.loadtxt(file_list, dtype=str)
    if filelist.ndim == 2:
        return filelist[:, 0]
    image_id = filelist[i].split("/")[-1].split(".")[0]
    return image_id

def multi_scale_predict(model, image_A, image_B, scales, num_classes, flip=False):
    H, W        = (image_A.size(2), image_A.size(3))
    upsize      = (ceil(H / 8) * 8, ceil(W / 8) * 8)
    upsample    = nn.Upsample(size=upsize, mode='bilinear', align_corners=True)
    pad_h, pad_w= upsize[0] - H, upsize[1] - W
    image_A     = F.pad(image_A, pad=(0, pad_w, 0, pad_h), mode='reflect')
    image_B     = F.pad(image_B, pad=(0, pad_w, 0, pad_h), mode='reflect')

    total_predictions = np.zeros((num_classes, image_A.shape[2], image_A.shape[3]))

    for scale in scales:
        scaled_img_A = F.interpolate(image_A, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_img_B = F.interpolate(image_B, scale_factor=scale, mode='bilinear', align_corners=False)
        scaled_prediction = upsample(model(A_l=scaled_img_A, B_l=scaled_img_B))
        
        if flip:
            fliped_img_A = scaled_img_A.flip(-1)
            fliped_img_B = scaled_img_B.flip(-1)
            fliped_predictions  = upsample(model(A_l=fliped_img_A, B_l=fliped_img_B))
            scaled_prediction   = 0.5 * (fliped_predictions.flip(-1) + scaled_prediction)
        total_predictions += scaled_prediction.data.cpu().numpy().squeeze(0)
    
    total_predictions /= len(scales)
    return total_predictions[:, :H, :W]

# Functions related to computing performance metrics for CD
def _update_metric(pred_l, target_l, sup_running_metric):
    """
    update metric
    """
    Gl_pred = torch.argmax(pred_l, 1)

    sup_current_score = sup_running_metric.update_cm(pr=Gl_pred.cpu().numpy(), gt=target_l.detach().cpu().numpy())
    
    return sup_current_score
    
# Collect the status of the epoch
def _collect_epoch_states(logs, sup_running_metric):
    sup_scores = sup_running_metric.get_scores()
    sup_epoch_acc = sup_scores['mf1']

    logs['sup_epoch_acc'] = sup_epoch_acc.item()

    for k, v in sup_scores.items():
        logs['sup_'+k] = v
            
    return logs

def _get_available_devices(gpu_ids):
    n_gpu = len(gpu_ids)
    if n_gpu == 0:
        return 'cpu'
    gpu_list = ','.join(str(x) for x in gpu_ids)
    os.environ['CUDA_VISIBLE_DEVICES'] = gpu_list
    device = torch.device('cuda' if n_gpu > 0 else 'cpu')
    return device

model_dict = {
    'cct': models.Consistency_ResNet50_CD,
    'gan': models.SemiCDNet_TGRS21
}

def main():
    args = parse_arguments()

    # CONFIG
    assert args.config
    config = json.load(open(args.config))
    scales = [1.0]
    num_classes = 2
    sup_running_metric = metrics.ConfuseMatrixMeter(n_class=num_classes)

    # DATA LOADER
    # config['val_loader']["batch_size"]  = 1
    # config['val_loader']["num_workers"] = 1
    # config['val_loader']["split"]       = "test"
    # config['val_loader']["shuffle"]     = False
    # config['val_loader']['data_dir']    = args.Dataset_Path
    loader = dataloaders.CDDataset(config['test_loader'])
    # palette     = get_voc_pallete(num_classes)

    # MODEL
    config['model']['supervised'] = True
    config['model']['semi'] = False
    model = model_dict[args.model](num_classes=num_classes, conf=config['model'], testing=True)
    print(f'\n{model}\n')
    assert config["resume_path"] is not None, "resume_path in config must not None"
        
    checkpoint = torch.load(config["resume_path"]+'_gen.pth')
    # model = torch.nn.DataParallel(model)
    try:
        print("Loading the state dictionery...")
        model.load_state_dict(checkpoint['state_dict'], strict=True)
    except Exception as e:
        print(f'Some modules are missing: {e}')
        model.load_state_dict(checkpoint['state_dict'], strict=False)
        
    device = _get_available_devices(config['gpu_ids'])
    model.eval()
    model.to(device)

    if args.save and not os.path.exists('outputs'):
        os.makedirs('outputs')
    
    #Set HTML
    web_dir = os.path.join(config["trainer"]["save_dir"],config["experim_name"], 'web')
    html_results = HTML(web_dir=web_dir, exp_name=config['experim_name']+"--Test--",
                            save_name=config['experim_name'], config=config)
    imgs_dir = os.path.join(config["trainer"]["save_dir"],config["experim_name"],'images')
    dir_exists(imgs_dir)
    dir_exists(os.path.join(config['trainer']['log_dir'], config['experim_name'], 'logs'))
    # LOOP OVER THE DATA
    tbar = tqdm(loader, ncols=100)
    # total_inter, total_union = 0, 0
    # total_correct, total_label = 0, 0
    setup_logger('test', os.path.join(config['trainer']['log_dir'], config['experim_name'], 'logs'),
                        'test', level=logging.INFO, screen=False)
    test_logger = logging.getLogger('test')
    test_logger.info('\n Begin Model Evaluation (testing).')
    
    for index, data in enumerate(tbar):
        image_A, image_B, label = data
        image_id = get_imgid_list(Dataset_Path=config['test_loader']['data_dir'], split=config['test_loader']["split"], i=index)
        image_A = image_A.cuda()
        image_B = image_B.cuda()
        label   = label.cuda()
        
        #PREDICT
        with torch.no_grad():
            # output = multi_scale_predict(model, image_A, image_B, scales, num_classes)
            output = model(image_A, image_B)
        _update_metric(output, label, sup_running_metric)
        prediction = np.asarray(torch.argmax(output, 1).cpu().numpy(), dtype=np.uint8)[0]
        
        #Calculate metrics
        # output = output.cuda()
        # label[label>=1] = 1
        # output = torch.unsqueeze(output, 0)
        # label  = torch.unsqueeze(label, 0)
        # correct, labeled, inter, union  = eval_metrics(output, label, num_classes)
        # total_inter, total_union        = total_inter+inter, total_union+union
        # total_correct, total_label      = total_correct+correct, total_label+labeled
        # pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
        # IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
        # tbar.set_description('Test Results | PixelAcc: {:.4f}, IoU(no-change): {:.4f}, IoU(change): {:.4f} |'.format(pixAcc, IoU[0], IoU[1]))

        #SAVE RESULTS
        # prediction_im = colorize_mask(prediction, palette)# 这里变化区域标红
        pred_L_show = torch.argmax(output, dim=1, keepdim=True).detach().cpu()
        L = label.unsqueeze(0).detach().cpu()
        tp = (L * pred_L_show)
        fp = (pred_L_show - tp)
        fn = L - tp
        tp = tp.repeat(1,3,1,1).permute(0,2,3,1)
        fp = fp.repeat(1,3,1,1).permute(0,2,3,1)
        fn = fn.repeat(1,3,1,1).permute(0,2,3,1)
        comp = torch.zeros_like(tp).float()
        comp[tp[:,:,:,0]==1] = torch.tensor([255.0,255.0,255.0]) #white
        comp[fp[:,:,:,0]==1] = torch.tensor([220.0,20.0,60.0])#red
        comp[fn[:,:,:,0]==1] = torch.tensor([0.0,0.0,205.0])# blue
        # comp = self.set_device(comp)
        comp = comp.squeeze(0).numpy()
        prediction_im = PIL.Image.fromarray(comp.astype(np.uint8)).convert('P')
        
        prediction_im.save(os.path.join(imgs_dir,image_id+'_comp.png'))
    
    #Printing average metrics on test-data
    # pixAcc = 1.0 * total_correct / (np.spacing(1) + total_label)
    # IoU = 1.0 * total_inter / (np.spacing(1) + total_union)
    # mIoU = IoU.mean()
    # seg_metrics = {"Pixel_Accuracy": np.round(pixAcc, 5), "Mean_IoU": np.round(mIoU, 5),
    #                             "Class_IoU": dict(zip(range(num_classes), np.round(IoU, 5)))}
    logs = { }
    logs = _collect_epoch_states(logs, sup_running_metric)
    logs['Mean_IoU'] = logs['sup_miou']
    logs['Pixel_Accuracy'] = logs['sup_epoch_acc']
    html_results.add_results(epoch=1, seg_resuts=logs)
    html_results.save()
    message = '[Test CD summary)]: mF1=%.5f \n' %\
                      (logs['sup_epoch_acc'])
    for k, v in logs.items():
        message += '{:s}: {:.4e} '.format(k, v) 
    message += '\n'
    test_logger.info(message)
    test_logger.info('End of testing...')
    

def parse_arguments():
    parser = argparse.ArgumentParser(description='PyTorch Training')
    parser.add_argument('-c', '--config', default='configs/config_LEVIR_50.json',type=str,
                        help='Path to the config file')
    parser.add_argument( '-m','--model', default='cct', type=str,)
    parser.add_argument( '-s', '--save', action='store_true', help='Save images')
    # parser.add_argument('-d', '--Dataset_Path', default="/data/datasets/LEVIR-CD/", type=str,
    #                     help='Path to dataset LEVIR-CD')
    args = parser.parse_args()
    return args

if __name__ == '__main__':
    main()

