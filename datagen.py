from albumentations import (
    PadIfNeeded,
    HorizontalFlip,
    VerticalFlip,    
    CenterCrop,    
    Crop,
    Compose,
    Transpose,
    RandomRotate90,
    ElasticTransform,
    GridDistortion, 
    OpticalDistortion,
    RandomSizedCrop,
    OneOf,
    CLAHE,
    RandomBrightnessContrast,    
    RandomGamma    
)
import os
import cv2
from config import param_maps
from tqdm import tqdm
from random import shuffle
import numpy as np


def image_resize(image, width = None, height = None, inter = cv2.INTER_AREA):
    dim = None
    (h, w) = image.shape[:2]

    if width is None and height is None:
        return image
    if width is None:
        r = height / float(h)
        dim = (int(w * r), height)

    else:
        r = width / float(w)
        dim = (width, int(h * r))

    resized = cv2.resize(image, dim, interpolation = inter)
    return resized


def flow(data_dir,Timage,Tmask,batch,size1,size2,augument=False): 


    images_ = os.listdir(data_dir+Timage)
    shuffle(images_)
    ids_int = list(range(len(images_)))
    NORMALIZE = 127.5
    while True:
        try:
            for start in range(0,len(ids_int),batch):
                x_batch = []
                y_batch = []
                end = min(start+batch,len(images_))
                batch_create = ids_int[start:end]
                jbs = dict()
                for loads in batch_create:
                    try:
                        img = cv2.imread(os.path.join(data_dir,Timage,images_[loads]))
                        img = image_resize(img,width=param_maps["scale"])
                        height_o_image,width_o_image = img.shape[0],img.shape[1]
                        if height_o_image % 2 != 0:
                            height_o_image = height_o_image-1
                        if width_o_image % 2 != 0:
                            width_o_image = width_o_image-1
                        jbs["width"] = width_o_image*2
                        jbs["height"] = height_o_image*2
                        img = cv2.resize(img,(width_o_image,height_o_image))
                        img = cv2.cvtColor(img,cv2.COLOR_BGR2RGB)
                        masks = cv2.imread(os.path.join(data_dir,Tmask,images_[loads]))
                        masks = cv2.resize(masks,(jbs["width"],jbs["height"]))
                    except:
                        continue
                    if augument:
                        aug = Compose([VerticalFlip(p=0.1),Transpose(p=0.01),RandomGamma(p=0.06),OpticalDistortion(p=0.00, distort_limit=0.7, shift_limit=0.3)])
                        augmented = aug(image=img, mask=masks)
                        img = augmented['image']
                        masks = augmented['mask']
                        x_batch.append(img)
                        y_batch.append(masks)
                    else:
                        x_batch.append(img)
                        y_batch.append(masks)
                x_batch = np.array(x_batch)/NORMALIZE
                x_batch = x_batch-1
                y_batch = np.array(y_batch)/NORMALIZE
                y_batch = y_batch-1
                yield x_batch,y_batch
                
        except:
            continue
