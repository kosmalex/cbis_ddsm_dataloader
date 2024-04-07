import os
import cv2 as cv
from PIL import Image

from torch.utils.data import Dataset
import torch
import pandas as pd
import numpy as np
from matplotlib import pyplot as plt
from transformers import SamProcessor
import random

class MAMOSAMDataset(Dataset):
  def __init__(self, dataframe, truncate, transforms=None):
    self.__dataframe   = dataframe
    self.transforms  = transforms
    self.__truncate    = truncate
    self.__curr_idx    = 0
    self.__train_mode  = False
    self.__test_mode   = False

    self.__use_clahe   = False
    self.__processor   = SamProcessor.from_pretrained("facebook/sam-vit-base")
    
    # Same as doing ([uint8_t, uint16_t, ...])(-1) in C
    self.__norm2tensor = lambda x, y: \
    torch.tensor(np.divide(np.tile(x, (1, 1, 1)).astype('float32'), 
                                      np.max([-1]).astype(y)))

  def __apply_clahe(self, x):
    image  = (x.squeeze(0) * 65535).numpy().astype('uint16')

    kernel = np.ones((5,5), np.float32)/25
    image  = cv.filter2D(image,-1,kernel)

    clahe = cv.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
    image = clahe.apply(image)
    
    return image

  def __get_bbox(self, mask, perturbation=20):
    if mask.max() == mask.min():
      return np.array([0, 0, 0, 0])

    y_indices, x_indices = np.where(mask > 0)
    
    x_min, x_max = np.min(x_indices), np.max(x_indices)
    y_min, y_max = np.min(y_indices), np.max(y_indices)
  
    # add perturbation to bounding box coordinates
    H, W = mask.shape
    x_min = max(0, x_min - random.randint(0, perturbation))
    x_max = min(W, x_max + random.randint(0, perturbation))
    y_min = max(0, y_min - random.randint(0, perturbation))
    y_max = min(H, y_max + random.randint(0, perturbation))
    bbox = np.array([x_min, y_min, x_max, y_max])

    return bbox

  def __getitem__(self, idx):
    is_slice = type(idx) == slice

    sub_df = self.__dataframe.iloc[idx] if is_slice else \
                                        pd.DataFrame(
                                          self.__dataframe.iloc[idx].to_dict(),
                                          index=[0]
                                        )

    result = []

    for i, row in sub_df.iterrows():
      image = cv.imread(row.image_path, cv.IMREAD_UNCHANGED)
      mask  = cv.imread(row.mask_path , cv.IMREAD_UNCHANGED)

      mask  = self.__norm2tensor(mask, 'uint8')
      image = self.__norm2tensor(image, 'uint16')
      packet_in = {
        "image_tensor_list": [image, mask],
        "item"             : row
      }
      packet_out = self.__truncate(packet_in)
      
      image = packet_out["image_tensor_list"][0]
      mask  = packet_out["image_tensor_list"][1]

      bimage = image

      rng_state = torch.get_rng_state()
      image = self.transforms(image)

      bmask = mask

      torch.set_rng_state(rng_state)
      mask  = self.transforms(mask)
      # print(mask.shape)
      mask  = mask[0].numpy()
      image = self.__apply_clahe(image)

      bbox  = self.__get_bbox(mask)

      mask  = cv.resize(mask, (256, 256))
      
      image = self.__norm2tensor(image, 'uint16')

      image = torch.tensor(np.tile(
        image, reps=(3, 1, 1)
      ))
      
      inputs = self.__processor(
        image,
        input_boxes    = torch.tensor(np.array([[[bbox]]])),
        do_rescale     = False,
        return_tensors = "pt"
      )

      inputs["mask"]  = torch.tensor(mask)
      inputs["image"] = image

      result.append({k:v.squeeze(0) for k, v in inputs.items()})

    if is_slice:
      return result

    return result[0]

  def __len__(self):
    return len(self.__dataframe.index)

  def __iter__(self):
    self.__curr_idx = 0
    return self

  def __next__(self):
    if self.__curr_idx < len(self):
      x = self[self.__curr_idx]
      self.__curr_idx += 1
      return x
    else:
      raise StopIteration

  def use_clahe(self, flag=True):
    self.__use_clahe = flag

  def set_processor(self, model_ckpt):
    self.__processor = SamProcessor.from_pretrained(model_ckpt)
