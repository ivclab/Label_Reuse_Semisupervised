import warnings
warnings.filterwarnings(action='ignore', category=FutureWarning)

from data import prepare_CIFAR10 
from sampler import PartitionedRepeatedShuffledSampler 
from criterion import CrossEntropyLoss 
from criterion import MatchingLoss 
from cache import Cache 
from procedure import run 

from wideresnet import WideResNet

import torch 
from torch import nn 
from torch import optim 
from torch.utils.data import DataLoader 
from torch.utils.tensorboard import SummaryWriter 

from tqdm import tqdm 

import argparse 

def parse_args(): 
    parser = argparse.ArgumentParser()  
    parser.add_argument('--dataset_root', type=str, default='./cifar10')
    parser.add_argument('--tensorboard_dir', type=str, default='./tensorboards')
    parser.add_argument('--batch_size', type=int, default=64) 
    parser.add_argument('--n_labeled', type=int, default=250) 
    parser.add_argument('--n_val', type=int, default=5000) 
    parser.add_argument('--k_augment', type=int, default=2) 
    parser.add_argument('--n_workers', type=int, default=4) 
    parser.add_argument('--pin_memory', action='store_true') 
    parser.add_argument('--output_device', type=int, default=0) 
    parser.add_argument('--n_partitions', type=int, default=1)
    parser.add_argument('--n_repeats', type=int, default=1) 
    parser.add_argument('--sparsity', type=int, default=10) 
    parser.add_argument('--rampup_steps', type=int, default=16384) 
    parser.add_argument('--alpha', type=float, default=0.75) 
    parser.add_argument('--T', type=float, default=0.5) 
    parser.add_argument('--ema_decay', type=float, default=0.999) 
    parser.add_argument('--lambda_u', type=float, default=75.0) 
    parser.add_argument('--n_classes', type=int, default=10)
    parser.add_argument('--n_update_imgs', type=int, default=1 << 16 << 10)
    parser.add_argument('--n_checkpoint_imgs', type=int, default=1 << 16) 
    parser.add_argument('--lr', type=float, default=2e-3)
    parser.add_argument('--weight_decay', type=float, default=2e-2)
    args, unknown = parser.parse_known_args() 
    return args 


if __name__ == '__main__': 
    print('[parse args]')
    args = parse_args() 
    print(args)

    print('[prepare data]') 
    labeledset, unlabeledset, valset, testset = prepare_CIFAR10(
        root=args.dataset_root, n_labeled=args.n_labeled , n_val=args.n_val, k_augment=args.k_augment
    ) 

    print('[init dataloaders]')
    labeledloader = DataLoader(
        dataset=labeledset, 
        batch_size=args.batch_size, 
        shuffle=True, 
        num_workers=args.n_workers, 
        drop_last=True, 
        pin_memory=args.pin_memory 
    ) 
    unlabeledloader = DataLoader(
        dataset=unlabeledset, 
        batch_size=args.batch_size, 
        sampler=PartitionedRepeatedShuffledSampler(
            n=len(unlabeledset), 
            n_partitions=args.n_partitions, 
            n_repeats=args.n_repeats, 
            batch_size=args.batch_size 
        ), 
        num_workers=args.n_workers, 
        drop_last=True, 
        pin_memory=args.pin_memory  
    ) 
    valloader = DataLoader(
        dataset=valset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.n_workers 
    )
    testloader = DataLoader(
        dataset=testset, 
        batch_size=args.batch_size, 
        shuffle=False, 
        num_workers=args.n_workers 
    )

    print('[init cache]')
    partition_size, _ = divmod(len(unlabeledloader.sampler), args.n_partitions) 
    assert _ == 0 
    cache = Cache(n_entries=partition_size, entry_size=args.sparsity).to(args.output_device) 

    print('[init model]')
    model = WideResNet(num_classes=args.n_classes).to(args.output_device) 
    model_ema = WideResNet(num_classes=args.n_classes).to(args.output_device) 
    model_ema.load_state_dict(model.state_dict()) 

    print('[init optimizer]')
    optimizer = optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.weight_decay) 

    print('[init critera]') 
    criterion_labeled = CrossEntropyLoss()  
    criterion_unlabeled = MatchingLoss()  
    criterion_val = nn.CrossEntropyLoss() 

    print('[start training]') 
    with SummaryWriter(log_dir=args.tensorboard_dir) as tblogger: 
        run(
            labeledloader=labeledloader, 
            unlabeledloader=unlabeledloader, 
            valloader=valloader,
            testloader=testloader, 
            model=model, 
            model_ema=model_ema, 
            optimizer=optimizer, 
            criterion_labeled=criterion_labeled,
            criterion_unlabeled=criterion_unlabeled, 
            criterion_val=criterion_val, 
            rampup_steps=args.rampup_steps, 
            cache=cache, 
            tblogger=tblogger, 
            args=args 
        )
    
    print('[done]')
