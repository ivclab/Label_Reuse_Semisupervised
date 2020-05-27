import numpy as np 

import torch 
from torch.utils.data import Dataset

import torchvision 
from torchvision import transforms 
from torchvision.datasets import CIFAR10 

from PIL import Image

TRANSFORM_CIFAR10_VAL = transforms.Compose([
    transforms.ToTensor(), 
    transforms.Normalize(mean=(0.4914, 0.4822, 0.4465), std=(0.2471, 0.2435, 0.2616))
])

TRANSFORM_CIFAR10_TRAIN = transforms.Compose([
    transforms.RandomCrop(
        size=(32, 32), 
        padding=4, 
        padding_mode='reflect'
    ), 
    transforms.RandomHorizontalFlip(p=0.5), 
    TRANSFORM_CIFAR10_VAL
]) 

def split_data(label, n_classes, n_labeled, n_val): 
    label = np.array(label)
    n_labeled_per_class, _ = divmod(n_labeled, n_classes) 
    n_val_per_class, _ = divmod(n_val, n_classes) 
    
    idx_labeled_train = [] 
    idx_unlabeled_train = [] 
    idx_val = [] 
    
    for class_label in range(n_classes): 
        idx, = np.nonzero(label == class_label) 
        np.random.shuffle(idx) 
        idx_labeled_train.extend(idx[:n_labeled_per_class]) 
        idx_unlabeled_train.extend(idx[n_labeled_per_class:-n_val_per_class]) 
        idx_val.extend(idx[-n_val_per_class:]) 
    np.random.shuffle(idx_labeled_train) 
    np.random.shuffle(idx_unlabeled_train) 
    np.random.shuffle(idx_val) 
    return idx_labeled_train, idx_unlabeled_train, idx_val 

class KAugment: 
    def __init__(self, transform, k): 
        self.k = k 
        self.transform = transform 
    def __call__(self, img): 
        return torch.stack([self.transform(img) for _ in range(self.k)], dim=0) 

class CIFAR10Labeled(CIFAR10): 
    def __init__(self, idx, root, train=True, transform=lambda x: x, target_transform=lambda x: x, download=True): 
        super(CIFAR10Labeled, self).__init__(
            root=root, 
            train=train, 
            transform=transform, 
            target_transform=target_transform, 
            download=download 
        ) 
        self.data = self.data[idx] 
        self.targets = np.array(self.targets)[idx] 
        
    def __getitem__(self, idx): 
        img = Image.fromarray(self.data[idx]) 
        target = self.targets[idx] 
        
        img = self.transform(img) 
        target = self.target_transform(target) 
        return img, target 
        
class CIFAR10Unlabeled(CIFAR10): 
    def __init__(self, idx, root, train=True, transform=lambda x: x, download=True): 
        super(CIFAR10Unlabeled, self).__init__(
            root=root, 
            train=train, 
            transform=transform, 
            target_transform=None, 
            download=download 
        ) 
        self.data = self.data[idx] 
        self.targets = np.array(self.targets)[idx] 
        
    def __getitem__(self, idx): 
        idx, idx_cache, update_needed = idx 
        
        img = Image.fromarray(self.data[idx]) 
        
        img = self.transform(img) 
        return img, idx_cache, update_needed
    
def prepare_CIFAR10(root, n_labeled, n_val, k_augment): 
    cifar10 = CIFAR10(root=root, train=True, download=True)  
    idx_labeled_train, idx_unlabeled_train, idx_val = split_data(label=cifar10.targets, n_classes=len(cifar10.class_to_idx), n_labeled=n_labeled, n_val=n_val) 
    labeledset = CIFAR10Labeled(
        idx=idx_labeled_train, 
        root=root, 
        train=True, 
        transform=TRANSFORM_CIFAR10_TRAIN 
    ) 
    unlabeledset = CIFAR10Unlabeled(
        idx=idx_unlabeled_train, 
        root=root, 
        train=True, 
        transform=KAugment(transform=TRANSFORM_CIFAR10_TRAIN, k=k_augment)  
    )
    valset = CIFAR10Labeled(
        idx=idx_val, 
        root=root, 
        train=True, 
        transform=TRANSFORM_CIFAR10_VAL 
    )
    testset = CIFAR10(
        root=root, 
        train=False, 
        transform=TRANSFORM_CIFAR10_VAL, 
        target_transform=None,  
        download=True
    ) 
    return labeledset, unlabeledset, valset, testset 

