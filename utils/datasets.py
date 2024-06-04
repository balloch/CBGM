
import os
import random

import numpy as np
import pandas as pd
from PIL import Image

import h5py
import matplotlib.pyplot as plt

import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import grad
from torchvision import transforms
from torchvision import datasets
import torchvision.datasets.utils as dataset_utils
from torchvision.utils import save_image


class ColoredMNIST(datasets.VisionDataset):
  """
  Colored MNIST dataset for testing IRM. Prepared using procedure from https://arxiv.org/pdf/1907.02893.pdf

  Args:
    root (string): Root directory of dataset where ``ColoredMNIST/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data',env='train', transform=None, target_transform=None):
    super(ColoredMNIST, self).__init__(root, transform=transform,
                                target_transform=target_transform)

    if env in ['train', 'test']:
      self.data_label_tuples = torch.load(os.path.join(self.root, 'ColoredMNIST', env) + '.pt')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, [target,color_red,color_green] = self.data_label_tuples[index]
    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, [target,color_red,color_green]

  def __len__(self):
    return len(self.data_label_tuples)


class Libero(datasets.VisionDataset):
  """
  Libero Dataset created from libero_90

  Args:
    root (string): Root directory of dataset where ``libero/*.pt`` will exist.
    env (string): Which environment to load. Must be 1 of 'train1', 'train2', 'test', or 'all_train'.
    transform (callable, optional): A function/transform that  takes in an PIL image
      and returns a transformed version. E.g, ``transforms.RandomCrop``
    target_transform (callable, optional): A function/transform that takes in the
      target and transforms it.
  """
  def __init__(self, root='./data',env='train', loaded_tensor=None, transform=None, target_transform=None):
    super(Libero, self).__init__(root, transform=transform,
                                target_transform=target_transform)
    if loaded_tensor:
      self.data_label_tuples = loaded_tensor
    else:
      if env in ['train', 'test']:
        self.data_label_tuples = torch.load(os.path.join(self.root, 'libero', env) + '.pt')

  def __getitem__(self, index):
    """
    Args:
        index (int): Index

    Returns:
        tuple: (image, target) where target is index of the target class.
    """
    img, [
        white_yellow_mug,
        butter,
        wine_bottle,
        yellow_book,
        ketchup,
        tomato_sauce,
        orange_juice,
        porcelain_mug,
        chefmate_8_frypan,
        cream_cheese,
        plate,
        chocolate_pudding,
        red_coffee_mug,
        moka_pot,
        basket,
        milk,
        white_bowl,
        wooden_tray,
        akita_black_bowl,
        alphabet_soup,
        black_book,
        new_salad_dressing] = self.data_label_tuples[index]
    if self.transform is not None:
      img = self.transform(img)

    if self.target_transform is not None:
      target = self.target_transform(target)

    return img, torch.tensor([
        white_yellow_mug,
        butter,
        wine_bottle,
        yellow_book,
        ketchup,
        tomato_sauce,
        orange_juice,
        porcelain_mug,
        chefmate_8_frypan,
        cream_cheese,
        plate,
        chocolate_pudding,
        red_coffee_mug,
        moka_pot,
        basket,
        milk,
        white_bowl,
        wooden_tray,
        akita_black_bowl,
        alphabet_soup,
        black_book,
        new_salad_dressing])

  def __len__(self):
    return len(self.data_label_tuples)


concept_dict = {
    'white_yellow_mug': 0,
    'butter': 1,
    'wine_bottle': 2,
    'yellow_book': 3,
    'ketchup': 4,
    'tomato_sauce': 5,
    'orange_juice': 6,
    'porcelain_mug': 7,
    'chefmate_8_frypan': 8,
    'cream_cheese': 9,
    'plate': 10,
    'chocolate_pudding': 11,
    'red_coffee_mug': 12,
    'moka_pot': 13,
    'basket': 14,
    'milk': 15,
    'white_bowl': 16,
    'wooden_tray': 17,
    'akita_black_bowl': 18,
    'alphabet_soup': 19,
    'black_book': 20,
    'new_salad_dressing': 21
}
train_set = []
test_set = []
train_files = []
test_files = []


def get_bddl(file):
    concept_list = []
    Lines = file.readlines()
    extract = False
    for line in Lines:
        line = line.strip()
        if line == ")":
            extract = False
        if extract:
            line = line.split('-')[-1].strip()
            concept_list.append(line)
        if line == "(:objects":
            extract = True

    arr = np.zeros(len(concept_dict.keys()))
    concept_list = [concept_dict[c] for c in concept_list]
    arr[concept_list] = 1
    return arr.tolist()


def create_libero_dataset(config, batch_size):
    for filename in os.listdir(config["dataset"]["img_dir"]):
        with h5py.File(config["dataset"]["img_dir"] + filename, "r") as img_file:
            filename = filename.split('_demo')[0] + ".bddl"
            bddl_file = open(config["dataset"]["bddl_dir"] + filename, 'r')
            concepts = get_bddl(bddl_file)
            rand = random.uniform(0, 1)
            for d in range(len(img_file['data'].keys())):
                data = list(img_file['data'][f'demo_{d}']['obs']['agentview_rgb'])
                for image in data:
                    img = Image.fromarray(image)
                    if rand < config["dataset"]["train_test_split"]:
                        train_set.append((img,concepts))
                        train_files.append(filename.split('.bddl')[0])
                    else:
                        test_set.append((img,concepts))
                        test_files.append(filename.split('.bddl')[0])


    if not os.path.exists('data/libero'):
        os.makedirs('data/libero')
    torch.save(train_set, 'data/libero/train.pt')
    torch.save(test_set, 'data/libero/test.pt')
    train_file_df = pd.DataFrame(train_files, columns=['file'])
    test_file_df = pd.DataFrame(test_files, columns=['file'])
    train_file_df.to_csv('data/libero/train_files.csv')
    test_file_df.to_csv('data/libero/test_files.csv')

    train_loader = torch.utils.data.DataLoader(
        Libero(loaded_tensor=train_set,
            transform=transforms.Compose([transforms.Resize(config["dataset"]["img_size"]),
                transforms.ToTensor(),
            ])),
        batch_size=batch_size,
        shuffle=True,
    )

    test_loader = torch.utils.data.DataLoader(
        Libero(loaded_tensor=test_set,
                transform=transforms.Compose([transforms.Resize(config["dataset"]["img_size"]),
                    transforms.ToTensor(),
                ])),
        batch_size=config["dataset"]["test_batch_size"],
        shuffle=True,
    )

    return train_loader, test_loader
