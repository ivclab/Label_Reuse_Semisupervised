import torch 
from torch.nn import functional as F 

import numpy as np 

from tqdm import tqdm 

def interleave_offsets(batch, nu):
    groups = [batch // (nu + 1)] * (nu + 1)
    for x in range(batch - sum(groups)):
        groups[-x - 1] += 1
    offsets = [0]
    for g in groups:
        offsets.append(offsets[-1] + g)
    assert offsets[-1] == batch
    return offsets

def interleave(xy, batch):
    nu = len(xy) - 1
    offsets = interleave_offsets(batch, nu)
    xy = [[v[offsets[p]:offsets[p + 1]] for p in range(nu + 1)] for v in xy]
    for i in range(1, nu + 1):
        xy[0][i], xy[i][i] = xy[i][i], xy[0][i]
    return [torch.cat(v, dim=0) for v in xy]

def get_interleave_idx(n, batch_size): 
    return torch.cat(interleave(torch.split(torch.arange(n), batch_size), batch_size)) 

def linear_rampup(idx_batch, n_update_batches): 
    return np.clip(idx_batch/n_update_batches, 0.0, 1.0) 

def sample_lambda(alpha): 
    beta = np.random.beta(alpha, alpha) 
    _lambda = max(beta, 1.0 - beta) 
    return _lambda 

def mixup(img, target, alpha): 
    idx_rand = torch.randperm(img.size(0)) 
    
    _lambda = sample_lambda(alpha) 
    img_mixed = _lambda * img + (1.0 - _lambda) * img[idx_rand] 
    target_mixed = _lambda * target + (1.0 - _lambda) * target[idx_rand] 
    return img_mixed, target_mixed 
  
def run(labeledloader, unlabeledloader, valloader, testloader, model, model_ema, optimizer, criterion_labeled, criterion_unlabeled, criterion_val, rampup_steps, cache, tblogger, args):    
    labelediter = iter(labeledloader) 
    unlabelediter = iter(unlabeledloader) 
    run_iterator = range(0, args.n_update_imgs, args.batch_size)
    
    for idx_batch, k_seen_imgs in enumerate(tqdm(run_iterator, desc='Train')): 
        model.train() 
        try: 
            img_labeled, label = next(labelediter) 
        except StopIteration:  
            labelediter = iter(labeledloader) 
            img_labeled, label = next(labelediter) 
        try: 
            img_unlabeled, idx_cache, update_needed = next(unlabelediter) 
        except StopIteration: 
            unlabelediter = iter(unlabeledloader) 
            img_unlabeled, idx_cache, update_needed = next(unlabelediter) 
        
        img_labeled = img_labeled.to(args.output_device) 
        label = label.to(args.output_device) 
        img_unlabeled = img_unlabeled.to(args.output_device) 
        
        target_x = F.one_hot(label, num_classes=args.n_classes).float()
        
        if torch.all(update_needed): 
            with torch.no_grad(): 
                logit = torch.stack([model(u) for u in torch.unbind(img_unlabeled, dim=1)], dim=1) 
            guessed = torch.mean(torch.softmax(logit, dim=-1), dim=1) 
            value_sparse, idx_sparse = torch.topk(guessed, k=args.sparsity, dim=-1, sorted=False) 
            heated = torch.pow(value_sparse, 1.0/args.T) 
            sharpened = heated / torch.sum(heated, dim=-1, keepdim=True)
            cache.write(idx=idx_cache, idx_sparse=idx_sparse, value_sparse=sharpened) 
        
        idx_sparse, value_sparse = cache.read(idx=idx_cache)
        
        target_u = torch.zeros(
            (args.batch_size, args.n_classes), 
            device=args.output_device
        ).scatter_(
            dim=-1, 
            index=idx_sparse, 
            src=value_sparse 
        )
        img_mixed, target_mixed = mixup(
            img=torch.cat([img_labeled, torch.flatten(img_unlabeled.transpose(0, 1), start_dim=0, end_dim=1)], dim=0), 
            target=torch.cat([target_x, target_u.repeat(args.k_augment, 1)], dim=0), 
            alpha=args.alpha 
        ) 
        
        idx_interleaved = get_interleave_idx(n=args.batch_size*(args.k_augment+1), batch_size=args.batch_size) 
        
        logit_mixed = torch.cat([model(chunk) for chunk in torch.chunk(img_mixed[idx_interleaved], chunks=args.k_augment+1, dim=0)], dim=0)[idx_interleaved] 
        logit_labeled, logit_unlabeled = torch.split(logit_mixed, split_size_or_sections=(args.batch_size, args.batch_size*args.k_augment), dim=0) 
        target_labeled, target_unlabeled = torch.split(target_mixed, split_size_or_sections=(args.batch_size, args.batch_size*args.k_augment), dim=0) 
        
        loss_labeled = criterion_labeled(logit_labeled, target_labeled)
        loss_unlabeled = criterion_unlabeled(logit_unlabeled, target_unlabeled) 
        loss = loss_labeled + linear_rampup(idx_batch, rampup_steps) * args.lambda_u * loss_unlabeled 
        
        optimizer.zero_grad() 
        loss.backward() 
        optimizer.step() 
        
        for p, p_ema in zip(model.parameters(), model_ema.parameters()): 
            p_ema.data.mul_(args.ema_decay).add_(p.data, alpha=1.0-args.ema_decay)
        
        checkpoint_step, checkpoint_indicator = divmod(k_seen_imgs, args.n_checkpoint_imgs) 
        if checkpoint_indicator == 0: 
            for b, b_ema in zip(model.buffers(), model_ema.buffers()): 
                b_ema.data.copy_(b.data) 
                
            model_ema.eval() 
            with torch.no_grad(): 
                correct_val = [] 
                for img_val, label_val in tqdm(valloader, desc='Val'): 
                    img_val = img_val.to(args.output_device) 
                    label_val = label_val.to(args.output_device) 
                    logit_val = model_ema(img_val) 
                    loss_val = criterion_val(logit_val, label_val) 
                    correct_val.extend((torch.argmax(logit_val, dim=-1) == label_val).cpu().numpy()) 
                acc_val = np.mean(correct_val) 
                
                correct_test = []
                for img_test, label_test in tqdm(testloader, desc='Test'): 
                    img_test = img_test.to(args.output_device) 
                    label_test = label_test.to(args.output_device) 
                    logit_test = model_ema(img_test) 
                    loss_text = criterion_val(logit_test, label_test) 
                    correct_test.extend((torch.argmax(logit_test, dim=-1) == label_test).cpu().numpy()) 
                acc_test = np.mean(correct_test) 
                
                tblogger.add_scalar('acc/acc_val', acc_val, checkpoint_step)
                tblogger.add_scalar('acc/acc_test', acc_test, checkpoint_step) 
                tblogger.add_scalar('linear_rampup', linear_rampup(idx_batch, rampup_steps), checkpoint_step)
                print(f'acc_val: {acc_val} | acc_test: {acc_test}')
