import torchvision.transforms as transforms
import torchvision, argparse
from torchvision import datasets, transforms
from torch.utils.data.sampler import SubsetRandomSampler
from torchvision.utils import save_image
import os, sys, time, math, os
from torchvision import transforms, utils, datasets
from torch.utils.data import Dataset, DataLoader, SubsetRandomSampler, WeightedRandomSampler
from torch.utils.data import random_split
from PIL import Image
import torch
import numpy as np
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.autograd import Variable
from itertools import product
import pandas as pd
import torchvision.models as models
from torch.optim.lr_scheduler import _LRScheduler
from torch.optim import lr_scheduler
from torchvision import datasets, transforms
#from pthflops import count_ops
from torch import Tensor
from typing import Callable, Any, Optional, List, Type, Union
import torch.nn.init as init
import functools
from tqdm import tqdm
from scipy.stats import entropy

def load_cifar_10(batch_size_train, batch_size_test, input_resize, split_rate=0.2):

	#To normalize the input images data.
	mean = [0.457342265910642, 0.4387686270106377, 0.4073427106250871]
	std =  [0.26753769276329037, 0.2638145880487105, 0.2776826934044154]

	# Note that we apply data augmentation in the training dataset.
	transformations_train = transforms.Compose([transforms.Resize(input_resize),
		transforms.CenterCrop(input_resize),
		transforms.RandomHorizontalFlip(p = 0.25),
		transforms.RandomRotation(25),
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),])

	# Note that we do not apply data augmentation in the test dataset.
	transformations_test = transforms.Compose([transforms.Resize(input_resize),
		transforms.CenterCrop(input_resize), 
		transforms.ToTensor(), 
		transforms.Normalize(mean = mean, std = std),
		])
  
	train_set = datasets.CIFAR10(root=".", train=True, download=True, transform=transformations_train)
	test_set = datasets.CIFAR10(root=".", train=False, download=True, transform=transformations_test)

	indices = np.arange(len(train_set))

	# This line defines the size of training dataset.
	train_size = int(len(indices) - int(split_rate*len(indices)))

	np.random.shuffle(indices)
	train_idx, val_idx = indices[:train_size], indices[train_size:]

	train_data = torch.utils.data.Subset(train_set, indices=train_idx)
	val_data = torch.utils.data.Subset(train_set, indices=val_idx)

	train_loader = torch.utils.data.DataLoader(train_data, batch_size=batch_size_train, shuffle=True, num_workers=4, pin_memory=True)
	val_loader = torch.utils.data.DataLoader(val_data, batch_size=batch_size_test, shuffle=False, num_workers=4, pin_memory=True)

	test_loader = DataLoader(test_set, batch_size_test, num_workers=4, pin_memory=True)

	return train_loader, val_loader, test_loader

def norm():
	norm_layer = [nn.ReLU(inplace=True), nn.MaxPool2d(kernel_size=3, stride=2), 
	nn.LocalResponseNorm(size=3, alpha=5e-05, beta=0.75)]
	return norm_layer

conv = lambda n: [nn.Conv2d(n, 32, kernel_size=3, stride=1, padding=1), nn.ReLU(inplace=True)]
cap =  lambda n, n_classes: [nn.MaxPool2d(kernel_size=3), Flatten(), nn.Linear(n, n_classes)]

class Branch(nn.Module):
	def __init__(self, layer):
		super(Branch, self).__init__()
		self.layer = nn.Sequential(*layer)
	def forward(self, x):
		return self.layer(x)

class Flatten(nn.Module):
	def __init__(self):
		super(Flatten, self).__init__()
    
	def forward(self, x):
		# Do your print / debug stuff here
		x = x.view(x.size(0), -1)
		return x

class B_AlexNet(nn.Module):
  def __init__(self, branch1, n_classes, inserted_layer, pretrained=True):
    super(B_AlexNet, self).__init__()
    self.stages = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.stage_id = 0

    backbone_model = models.alexnet(pretrained=pretrained)
    backbone_model_features = backbone_model.features

    for i, layer in enumerate(backbone_model_features):
      self.layers.append(layer)
      #print(i, layer)
      if(i == inserted_layer):
        self.add_exit_point(branch1)
    
    self.layers.append(backbone_model.avgpool)    
    self.stages.append(nn.Sequential(*self.layers))
    del self.layers   
    self.classifier = backbone_model.classifier
    #self.classifier[1] = nn.Linear(9216, 4096)
    #self.classifier[4] = nn.Linear(4096, 1024)
    self.classifier[6] = nn.Linear(4096, n_classes)    
    self.softmax = nn.Softmax(dim=1)

  def add_exit_point(self, branch1):
    self.stages.append(nn.Sequential(*self.layers))
    self.exits.append(nn.Sequential(*branch1))
    self.layers = nn.ModuleList()
    self.stage_id += 1    

  def forwardMain(self, x):
    for i, stage in enumerate(self.exits):
      x = self.stages[i](x)

    x = self.stages[-1](x)    
    x = torch.flatten(x, 1)

    output = self.classifier(x)
    conf, infered_class = torch.max(self.softmax(output), 1)
    return output, conf, infered_class

  def forward(self, x):
    n_exits = len(self.exits)
    output_list, conf_list, class_list = [], [], []

    for i, exit in enumerate(self.exits):
      x = self.stages[i](x)
      output_branch = self.exits[i](x)
      conf, infered_class = torch.max(self.softmax(output_branch), 1)
      output_list.append(output_branch), conf_list.append(conf), class_list.append(infered_class)
    
    x = self.stages[-1](x)    
    x = torch.flatten(x, 1)

    #print(x.shape)
    #print(self.classifier)
    output = self.classifier(x)
    output_list.append(output)
    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf), class_list.append(infered_class)

    return output_list, conf_list, class_list

def build_b_alexnet(device, n_classes, inserted_layer, pretrained=True):
  branch1 = norm() + conv(64) + conv(32) + cap(512, n_classes)
  b_alexnet = B_AlexNet(branch1, n_classes, inserted_layer)
  b_alexnet = b_alexnet.to(device)

  return b_alexnet


def save_results(result, save_path):
	df_result = pd.read_csv(save_path) if (os.path.exists(save_path)) else pd.DataFrame()
	df = pd.DataFrame(np.array(list(result.values())).T, columns=list(result.keys()))
	df_result = df_result.append(df)
	df_result.to_csv(save_path)


def expCollectingData(model, test_loader, device, n_branches):
	df_result = pd.DataFrame()
	n_exits = n_branches + 1
	conf_branches_list, infered_class_branches_list, target_list = [], [], []
	correct_list, id_list = [], []

	model.eval()

	test_dataset_size = len(test_loader)

	with torch.no_grad():
		for i, (data, target) in tqdm(enumerate(test_loader, 1)):
			
			#print("Image id: %s/%s"%(i, test_dataset_size))
			data, target = data.to(device).float(), target.to(device)

			_, conf_branches, infered_class_branches = model(data)

			conf_branches_list.append([conf.item() for conf in conf_branches])
			infered_class_branches_list.append([inf_class.item() for inf_class in infered_class_branches])    
			correct_list.append([infered_class_branches[i].eq(target.view_as(infered_class_branches[i])).sum().item() for i in range(n_exits)])
			id_list.append(i), target_list.append(target.item())

			del data, target
			torch.cuda.empty_cache()

	conf_branches_list = np.array(conf_branches_list)
	infered_class_branches_list = np.array(infered_class_branches_list)
	correct_list = np.array(correct_list)

	results = {"target": target_list, "id": id_list}

	for i in range(n_exits):
		results.update({"conf_branch_%s"%(i+1): conf_branches_list[:, i],
			"infered_class_branches_%s"%(i+1): infered_class_branches_list[:, i],
			"correct_branch_%s"%(i+1): correct_list[:, i]})

	return results


parser = argparse.ArgumentParser(description="Evaluating early-exits DNNs perfomance")

parser.add_argument('--model_id', type=int, help='Model id')

args = parser.parse_args()


n_classes = 10
inserted_layer = 2
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
input_dim = 224
batch_size_train = 256
batch_size_test = 1
n_exits = 1
split_ratio = 0.2
pretrained = True
n_branches = 1

root_path = "."
model_save_path = os.path.join(root_path, "ee_alexnet_testing_for_ucb_%s.pth"%(args.model_id))
expSavePath = os.path.join(root_path, "results_ee_alexnet_for_ucb_%s.csv"%(args.model_id))

train_loader, val_loader, test_loader = load_cifar_10(batch_size_train, batch_size_test, input_dim, split_ratio)

early_exit_dnn = build_b_alexnet(device, n_classes, inserted_layer, pretrained)
early_exit_dnn.load_state_dict(torch.load(model_save_path, map_location=device)["model_state_dict"])


result = expCollectingData(early_exit_dnn, test_loader, device, n_branches)
save_results(result, expSavePath)
