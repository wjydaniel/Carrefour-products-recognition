import argparse
#import better_exceptions
from pathlib import Path
import torch
import torch.nn.parallel
import torch.backends.cudnn as cudnn
import torch.optim
import torch.utils.data
from torch.utils.data import DataLoader
import pretrainedmodels
import pretrainedmodels.utils
from model import get_model
from dataset import ProductDataset
from defaults import _C as cfg
from train import validate
import os
import numpy as np
from tqdm import tqdm
import pandas as pd

def get_args():
    model_names = sorted(name for name in pretrainedmodels.__dict__
                         if not name.startswith("__")
                         and name.islower()
                         and callable(pretrainedmodels.__dict__[name]))
    parser = argparse.ArgumentParser(description=f"available models: {model_names}",
                                     formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True, help="Data root directory")
    parser.add_argument("--resume", type=str, required=True, help="Model weight to be tested")
    parser.add_argument("opts", default=[], nargs=argparse.REMAINDER,
                        help="Modify config options using the command-line")
    args = parser.parse_args()
    return args


def main():
    args = get_args()

    if args.opts:
        cfg.merge_from_list(args.opts)

    cfg.freeze()

    # create model
    print("=> creating model '{}'".format(cfg.MODEL.ARCH))
    model = get_model(model_name=cfg.MODEL.ARCH, pretrained=None)
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)

    # load checkpoint
    resume_path = args.resume

    if Path(resume_path).is_file():
        print("=> loading checkpoint '{}'".format(resume_path))
        checkpoint = torch.load(resume_path, map_location="cpu")
        model.load_state_dict(checkpoint['state_dict'])
        print("=> loaded checkpoint '{}'".format(resume_path))
    else:
        raise ValueError("=> no checkpoint found at '{}'".format(resume_path))

    if device == "cuda":
        cudnn.benchmark = True

    
    
    #preds = []
    #gt = []
    test_list_1 = []
    for i,j,k in os.walk('./gdrive/MyDrive/data/26953_Bébé'):
      
      test_list_1 = np.char.add(np.array([args.data_dir + '/26953_Bébé/']*len(k)), k)
      break
    
    for i,j,k in os.walk(args.data_dir +'/2112_Epicerie_salee'):
      test_list_2 = np.char.add(np.array([args.data_dir + '/2112_Epicerie_salee/']*len(k)), k)
      break
    #test_dataset1 = ProductDataset(args.data_dir, "all", img_size=cfg.MODEL.IMG_SIZE, augment=False,name_list=test_list_1)
    #test_loader1 = DataLoader(test_dataset1, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
    #                         num_workers=cfg.TRAIN.WORKERS, drop_last=False)
    test_dataset2 = ProductDataset(args.data_dir, "all", img_size=cfg.MODEL.IMG_SIZE, augment=False,name_list=test_list_2)
    test_loader2 = DataLoader(test_dataset2, batch_size=cfg.TEST.BATCH_SIZE, shuffle=False,
                             num_workers=cfg.TRAIN.WORKERS, drop_last=False)
    model.eval()
    #loss_monitor = AverageMeter()
    #accuracy_monitor = AverageMeter()
    true_label = []
    predicted_label = []
    with torch.no_grad():
        with tqdm(test_loader2) as _tqdm:
            for i, (x, y) in enumerate(_tqdm):
                x = x.to(device)
                y = y.to(device)

                # compute output
                _,predicted = model(x).max(1)
                true_label += y.tolist()
                predicted_label += predicted.tolist()

                correct_num = predicted.eq(y).sum().item()
                sample_num = x.size(0)
                #loss_monitor.update(cur_loss, sample_num)
                #accuracy_monitor.update(correct_num, sample_num)
                

    pd.DataFrame(predicted_label).to_csv('./gdrive/MyDrive/data/pred2.csv')
    pd.DataFrame(true_label).to_csv('./gdrive/MyDrive/data/true2.csv')

if __name__ == '__main__':
    main()
