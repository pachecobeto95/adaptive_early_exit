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
from pthflops import count_ops


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


class EarlyExitBlock(nn.Module):
  """
  This EarlyExitBlock allows the model to terminate early when it is confident for classification.
  """
  def __init__(self, input_shape, pool_size, n_classes, exit_type, device):
    super(EarlyExitBlock, self).__init__()
    self.input_shape = input_shape

    _, channel, width, height = input_shape
    self.expansion = width * height if exit_type == 'plain' else 1

    self.layers = nn.ModuleList()

    if (exit_type == 'bnpool'):
      self.layers.append(nn.BatchNorm2d(channel))

    if (exit_type != 'plain'):
      self.layers.append(nn.AdaptiveAvgPool2d(pool_size))
    
    #This line defines the data shape that fully-connected layer receives.
    current_channel, current_width, current_height = self.get_current_data_shape()

    self.layers = self.layers#.to(device)

    #This line builds the fully-connected layer
    self.classifier = nn.Sequential(nn.Linear(current_channel*current_width*current_height, n_classes))#.to(device)

    self.softmax_layer = nn.Softmax(dim=1)


  def get_current_data_shape(self):
    _, channel, width, height = self.input_shape
    temp_layers = nn.Sequential(*self.layers)

    input_tensor = torch.rand(1, channel, width, height)
    _, output_channel, output_width, output_height = temp_layers(input_tensor).shape
    return output_channel, output_width, output_height
        
  def forward(self, x):
    for layer in self.layers:
      x = layer(x)
    x = x.view(x.size(0), -1)
    output = self.classifier(x)
    #confidence = self.softmax_layer()
    return output



class Early_Exit_AlexNet(nn.Module):
  def __init__(self, n_classes: int, 
               pretrained: bool, n_branches: int, input_shape:tuple, 
               exit_type: str, device, distribution="linear"):
    super(Early_Exit_AlexNet, self).__init__()

    """
    This classes builds an early-exit DNNs architectures
    Args:

    model_name: model name 
    n_classes: number of classes in a classification problem, according to the dataset
    pretrained: 
    n_branches: number of branches (early exits) inserted into middle layers
    input_shape: shape of the input image
    exit_type: type of the exits
    distribution: distribution method of the early exit blocks.
    device: indicates if the model will processed in the cpu or in gpu
    
    Note: the term "backbone model" refers to a regular DNN model, considering no early exits.

    """
    self.n_classes = n_classes
    self.pretrained = pretrained
    self.n_branches = n_branches
    self.input_shape = input_shape
    self.exit_type = exit_type
    self.distribution = distribution
    self.device = device
    self.channel, self.width, self.height = input_shape
    self.pool_size = 1

    self.early_exit_alexnet()

  def select_distribution_method(self):
    """
    This method selects the distribution method to insert early exits into the middle layers.
    """
    distribution_method_dict = {"linear":self.linear_distribution,
                                "pareto":self.paretto_distribution,
                                "fibonacci":self.fibo_distribution}
    return distribution_method_dict.get(self.distribution, self.invalid_distribution)
    
  def linear_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a linear distribution.
    """
    flop_margin = 1.0 / (self.n_branches+1)
    return self.total_flops * flop_margin * (i+1)

  def paretto_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a pareto distribution.
    """
    return self.total_flops * (1 - (0.8**(i+1)))

  def fibo_distribution(self, i):
    """
    This method defines the Flops to insert an early exits, according to a fibonacci distribution.
    """
    gold_rate = 1.61803398875
    return total_flops * (gold_rate**(i - self.num_ee))

  def verifies_nr_exits(self, backbone_model):
    """
    This method verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    """
    
    total_layers = len(list(backbone_model.children()))
    if (self.n_branches >= total_layers):
      raise Exception("The number of early exits is greater than number of layers in the DNN backbone model.")

  def countFlops(self, model):
    """
    This method counts the numper of Flops in a given full DNN model or intermediate DNN model.
    """
    input = torch.rand(1, self.channel, self.width, self.height)#.to(self.device)
    flops, all_data = count_ops(model, input, print_readable=False, verbose=False)
    return flops

  def where_insert_early_exits(self):
    """
    This method defines where insert the early exits, according to the dsitribution method selected.
    Args:

    total_flops: Flops of the backbone (full) DNN model.
    """
    threshold_flop_list = []
    distribution_method = self.select_distribution_method()

    for i in range(self.n_branches):
      threshold_flop_list.append(distribution_method(i))

    return threshold_flop_list

  def invalid_model(self):
    raise Exception("This DNN model has not implemented yet.")
  def invalid_distribution(self):
    raise Exception("This early-exit distribution has not implemented yet.")

  def is_suitable_for_exit(self):
    """
    This method answers the following question. Is the position to place an early exit?
    """
    intermediate_model = nn.Sequential(*(list(self.stages)+list(self.layers)))
    x = torch.rand(1, 3, 224, 224)#.to(self.device)
    current_flop, _ = count_ops(intermediate_model, x, verbose=False, print_readable=False)
    return self.stage_id < self.n_branches and current_flop >= self.threshold_flop_list[self.stage_id]

  def add_exit_block(self):
    """
    This method adds an early exit in the suitable position.
    """
    input_tensor = torch.rand(1, self.channel, self.width, self.height)

    self.stages.append(nn.Sequential(*self.layers))
    x = torch.rand(1, 3, 224, 224)#.to(self.device)
    feature_shape = nn.Sequential(*self.stages)(x).shape
    self.exits.append(EarlyExitBlock(feature_shape, self.pool_size, self.n_classes, self.exit_type, self.device))#.to(self.device))
    self.layers = nn.ModuleList()
    self.stage_id += 1    

  def set_device(self):
    """
    This method sets the device that will run the DNN model.
    """

    self.stages.to(self.device)
    self.exits.to(self.device)
    self.layers.to(self.device)
    self.classifier.to(self.device)

  def set_device_resnet50(self):
    self.stages.to(self.device)
    self.exits.to(self.device)
    self.layers.to(self.device)
    self.classifier.to(self.device)

  def early_exit_alexnet(self):
    """
    This method inserts early exits into a Alexnet model
    """

    self.stages = nn.ModuleList()
    self.exits = nn.ModuleList()
    self.layers = nn.ModuleList()
    self.cost = []
    self.stage_id = 0

    # Loads the backbone model. In other words, Alexnet architecture provided by Pytorch.
    backbone_model = models.alexnet(self.pretrained)

    # It verifies if the number of early exits provided is greater than a number of layers in the backbone DNN model.
    #self.verifies_nr_exit_alexnet(backbone_model.features)
    
    # This obtains the flops total of the backbone model
    self.total_flops = self.countFlops(backbone_model)

    # This line obtains where inserting an early exit based on the Flops number and accordint to distribution method
    self.threshold_flop_list = self.where_insert_early_exits()

    for layer in backbone_model.features:
      self.layers.append(layer)
      if (isinstance(layer, nn.ReLU)) and (self.is_suitable_for_exit()):
        self.add_exit_block()

    
    
    self.layers.append(nn.AdaptiveAvgPool2d(output_size=(6, 6)))
    self.stages.append(nn.Sequential(*self.layers))

    
    self.classifier = backbone_model.classifier
    self.classifier[6] = nn.Linear(in_features=4096, out_features=self.n_classes, bias=True)
    self.softmax = nn.Softmax(dim=1)
    self.set_device()


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

    output = self.classifier(x)
    output_list.append(output)
    conf, infered_class = torch.max(self.softmax(output), 1)
    conf_list.append(conf), class_list.append(infered_class)

    return output_list, conf_list, class_list



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
distribution = "linear"
exit_type = "bnpool"
input_shape = (3, input_dim, input_dim)


root_path = "."
model_save_path = os.path.join(root_path, "ee_alexnet_testing_for_ucb_%s.pth"%(args.model_id))
expSavePath = os.path.join(root_path, "results_ee_alexnet_for_ucb_%s.csv"%(args.model_id))

train_loader, val_loader, test_loader = load_cifar_10(batch_size_train, batch_size_test, input_dim, split_ratio)

early_exit_dnn = Early_Exit_AlexNet(n_classes, pretrained, n_branches, input_shape, exit_type, device, distribution)
early_exit_dnn.load_state_dict(torch.load(model_save_path, map_location=device)["model_state_dict"])
early_exit_dnn = early_exit_dnn.to(device)


result = expCollectingData(early_exit_dnn, test_loader, device, n_branches)
save_results(result, expSavePath)