import argparse
#import better_exceptions
from pathlib import Path
import numpy as np
import pandas as pd
import torch
import cv2
from torch.utils.data import Dataset
from imgaug import augmenters as iaa
import os
from tqdm import tqdm



class ImgAugTransform:
    def __init__(self):
        self.aug = iaa.Sequential([
            iaa.OneOf([
                iaa.Sometimes(0.25, iaa.AdditiveGaussianNoise(scale=0.1 * 255)),
                iaa.Sometimes(0.25, iaa.GaussianBlur(sigma=(0, 3.0)))
                ]),
            iaa.Affine(
                rotate=(-20, 20), mode="edge",
                scale={"x": (0.95, 1.05), "y": (0.95, 1.05)},
                translate_percent={"x": (-0.05, 0.05), "y": (-0.05, 0.05)}
            ),
            iaa.AddToHueAndSaturation(value=(-10, 10), per_channel=True),
            iaa.GammaContrast((0.3, 2)),
            iaa.Fliplr(0.5),
        ])

    def __call__(self, img):
        img = np.array(img)
        img = self.aug.augment_image(img)
        return img


class ProductDataset(Dataset):
    def __init__(self, data_dir, class_name, img_size=224, augment=False, name_list=[]):
        print('=> creating dataset' + ' ' + class_name)
      ######## this part reserved to Bebe #############################################
        assert(class_name in ("26953_Bébé",'2112_Epicerie_salee','all','test'))
        if class_name == "26953_Bébé":
          level3_name = ['26970', '26974', '26971', '27062', '26975', '26976', '26979', '26978',
                '27065', '27061', '26964', '27064', '27063', '26967', '26963', '26962',
                '26972', '26977', '27060', '26986', '26973', '27059', '26980', '26997',
                '27066', '26996']
        elif class_name == '2112_Epicerie_salee':
          level3_name = ['2181', '2115', '2117', '2116', '2119', '2123', '2124', '2122',
                '2127', '2131', '2128', '2129', '2133', '2136', '2142', '2137',
                '2135', '2138', '2140', '2141', '2134', '2154', '2144', '2156',
                '2150', '2158', '2159', '2160', '2161', '2168', '2172', '2164',
                '2166', '2167', '2165', '2175', '2174', '2176', '2177', '2182',
                '2114', '2125', '2179', '2130', '2139', '2155', '2163', '2180', '25997']
        else:
          level3_name = ['26970', '26974', '26971', '27062', '26975', '26976', '26979', '26978',
                '27065', '27061', '26964', '27064', '27063', '26967', '26963', '26962',
                '26972', '26977', '27060', '26986', '26973', '27059', '26980', '26997',
                '27066', '26996'] + ['2181', '2115', '2117', '2116', '2119', '2123', '2124', '2122',
                '2127', '2131', '2128', '2129', '2133', '2136', '2142', '2137',
                '2135', '2138', '2140', '2141', '2134', '2154', '2144', '2156',
                '2150', '2158', '2159', '2160', '2161', '2168', '2172', '2164',
                '2166', '2167', '2165', '2175', '2174', '2176', '2177', '2182',
                '2114', '2125', '2179', '2130', '2139', '2155', '2163', '2180', '25997']
        level3_class = list(range(len(level3_name)))
        level3_dic = dict(zip(level3_name, level3_class))
      #######################################################################################  
        csv_path = Path(data_dir).joinpath("data.csv")
        #if class_name=='all':
        #  img_dir = [Path(data_dir).joinpath('26953_Bébé'),Path(data_dir).joinpath('2112_Epicerie_salée','all')]
        #else:
        #  img_dir = Path(data_dir).joinpath(class_name)

        self.img_size = img_size
        self.augment = augment

        if augment:
            self.transform = ImgAugTransform()
        else:
            self.transform = lambda i: i

        self.x = []
        self.y = []
        self.std = []
        df = pd.read_csv(str(csv_path))
        
        ignore_img_names = pd.read_csv(Path(data_dir).joinpath('ignore.csv')).values[:,1]
        
        ruin_img_num = 0
        
        
        for img_name in tqdm(name_list):
            b,c,d = img_name.split('_')[-3:]
            
            if 'image_'+b+'_'+c+'_'+d in ignore_img_names: 
              ruin_img_num += 1
              continue

            barcode = int(b)
            
            level3 = df[df['barcode']==barcode]['nodeid3']
            if level3.isna().values:
                ignore_img_names.append(img_name)
                continue
            
            #img_path = img_dir.joinpath(img_name)
            img_path = Path(img_name)
            assert(img_path.is_file())
            
            self.x.append(str(img_path))
            self.y.append(level3_dic[str(int(level3.values))])
        print('ruin images nums = ',ruin_img_num) 

    def __len__(self):
        return len(self.y)

    def __getitem__(self, idx):
        img_path = self.x[idx]
        age = self.y[idx]

        img = cv2.imread(str(img_path))

        img = cv2.resize(img, (self.img_size, self.img_size))
      
        img = self.transform(img).astype(np.float32)
        return torch.from_numpy(np.transpose(img, (2, 0, 1))), np.clip(round(age), 0, 25+49)


def main():
    parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument("--data_dir", type=str, required=True)
    parser.add_argument("--data_type", type=str, required=True)
    args = parser.parse_args()

    for i,j,k in os.walk(args.data_dir + args.data_type):
      name_list = k
      break
    train_list = np.random.choice(name_list,int(len(name_list)*0.8),False)
    val_list = np.setdiff1d(name_list,train_list,True)
    
    dataset = FaceDataset(args.data_dir, args.data_type, name_list=train_list)
    print("26953_Bébé train dataset len: {}".format(len(dataset)))
    dataset = FaceDataset(args.data_dir, args.data_type, name_list=val_list)
    print("26953_Bébé validation dataset len: {}".format(len(dataset)))


if __name__ == '__main__':
  
  main()
