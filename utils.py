import torch
import math
import os
import torch.distributed as dist
from collections import OrderedDict

class WarmupCosineSchedule(object):

    def __init__(
        self,
        optimizer,
        warmup_steps,
        start_lr,
        ref_lr,
        T_max,
        last_epoch=-1,
        final_lr=0.
    ):
        self.optimizer = optimizer
        self.start_lr = start_lr
        self.ref_lr = ref_lr
        self.final_lr = final_lr
        self.warmup_steps = warmup_steps
        self.T_max = T_max - warmup_steps
        self._step = 0.

    def step(self):
        self._step += 1
        if self._step < self.warmup_steps:
            progress = float(self._step) / float(max(1, self.warmup_steps))
            new_lr = self.start_lr + progress * (self.ref_lr - self.start_lr)
        else:
            # -- progress after warmup
            progress = float(self._step - self.warmup_steps) / float(max(1, self.T_max))
            new_lr = max(self.final_lr,
                         self.final_lr + (self.ref_lr - self.final_lr) * 0.5 * (1. + math.cos(math.pi * progress)))

        for group in self.optimizer.param_groups:
            group['lr'] = new_lr

        return new_lr

class CosineWDSchedule(object):

    def __init__(
        self,
        optimizer,
        ref_wd,
        T_max,
        final_wd=0.
    ):
        self.optimizer = optimizer
        self.ref_wd = ref_wd
        self.final_wd = final_wd
        self.T_max = T_max
        self._step = 0.

    def step(self):
        self._step += 1
        progress = self._step / self.T_max
        new_wd = self.final_wd + (self.ref_wd - self.final_wd) * 0.5 * (1. + math.cos(math.pi * progress))

        if self.final_wd <= self.ref_wd:
            new_wd = max(self.final_wd, new_wd)
        else:
            new_wd = min(self.final_wd, new_wd)

        for group in self.optimizer.param_groups:
            if ('WD_exclude' not in group) or not group['WD_exclude']:
                group['weight_decay'] = new_wd
        return new_wd

def pretrain_optimizer(encoder, predictor, betas, epsilon, zero_init_bias_wd):
    param_groups = [
        {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in encoder.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': zero_init_bias_wd,
            'weight_decay': 0,
        }, {
            'params': (p for n, p in predictor.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': zero_init_bias_wd,
            'weight_decay': 0,
        },
    ]

    print('Using AdamW')
    betas=(0.9, 0.999)
    eps=1e-8
    optimizer = torch.optim.AdamW(param_groups, betas=betas, eps=eps)
    return optimizer


def save_checkpoint(rank, encoder, predictor, optimizer, scaler, target_encoder, epoch, batch_size, world_size, lr, path):
    if rank != 0:
        print("Returning from checkpoint")
        return
    save_dict = {
        'encoder': encoder.state_dict(),
        'predictor': predictor.state_dict(),
        'opt': optimizer.state_dict(),
        'scaler': None if scaler is None else scaler.state_dict(),
        'target_encoder': target_encoder.state_dict(),
        'epoch': epoch,
        #'loss': loss_meter.avg,
        'batch_size': batch_size,
        'world_size': world_size,
        'lr': lr,
    }
    torch.save(save_dict, path)


def save_snapshot(rank, encoder, predictor, optimizer, scaler, target_encoder, epoch, batch_size, world_size, lr, path):
    save_dict = {
        'ENCODER': encoder.state_dict(),
        'PREDICTOR': predictor.state_dict(),
        'OPTIMIZER': optimizer.state_dict(),
        'SCALER': None if scaler is None else scaler.state_dict(),
        'TARGET_ENCODER': target_encoder.state_dict(),
        'EPOCH': epoch,
        #'loss': loss_meter.avg,
        'BATCH_SIZE': batch_size,
        'WORLD_SIZE': world_size,
        'LEARNING_RATE': lr,
    }
    torch.save(save_dict, path)

def load_checkpoint(
    r_path,
    encoder,
    predictor,
    target_encoder,
    opt,
    scaler,
    device):
    
    checkpoint = torch.load(r_path, map_location=torch.device(device))  # Load the checkpoint (adjust map_location as needed)
    print("checkpoint", checkpoint)
    # Adjust the state_dict keys by removing the 'module.' prefix
    # state_dict = checkpoint['state_dict'] if 'state_dict' in checkpoint else checkpoint
    # print("state_dict", state_dict)
    # new_state_dict = {key.replace("module.", ""): value for key, value in state_dict.items()}
    # checkpoint1 = new_state_dict
    # print("checkpoint1", checkpoint1)

    epoch = checkpoint['epoch']

    # -- loading encoder
    pretrained_dict = checkpoint['encoder']
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
    # print("Examine pretrained_dict", pretrained_dict)
    # print("Examine new_dict", new_state_dict)
    msg = encoder.load_state_dict(new_state_dict)
    #logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    # -- loading predictor
    pretrained_dict = checkpoint['predictor']
    new_state_dict = {}
    for k, v in pretrained_dict.items():
        name = k[7:] # remove 'module.' of dataparallel
        new_state_dict[name]=v
    msg = predictor.load_state_dict(new_state_dict)

    # -- loading optimizer
    opt.load_state_dict(checkpoint['opt'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])

    # -- loading target_encoder
    if target_encoder is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint['target_encoder']
        new_state_dict = {}
        for k, v in pretrained_dict.items():
            name = k[7:] # remove 'module.' of dataparallel
            new_state_dict[name]=v
        msg = target_encoder.load_state_dict(new_state_dict)
    
    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )

def load_checkpoint_finetune(checkpoint_path, classifier, opt, scaler):
    if not os.path.exists(checkpoint_path):
        raise FileNotFoundError(f"Checkpoint file {checkpoint_path} not found.")

    print(f"Loading checkpoint from {checkpoint_path}...")
    checkpoint = torch.load(checkpoint_path, map_location='cpu')

    # Restore classifier
    classifier.load_state_dict(checkpoint['classifier'])
    print("Classifier state restored.")

    # Restore optimizer
    opt.load_state_dict(checkpoint['opt'])
    print("Optimizer state restored.")

    # Restore scaler (if applicable)
    scaler = None
    if 'scaler' in checkpoint and checkpoint['scaler'] is not None:
        scaler = torch.cuda.amp.GradScaler()
        scaler.load_state_dict(checkpoint['scaler'])
        print("Scaler state restored.")

    # Restore other training states
    epoch = checkpoint.get('epoch', 0)
    batch_size = checkpoint.get('batch_size', None)
    world_size = checkpoint.get('world_size', None)
    lr = checkpoint.get('lr', None)

    print(f"Checkpoint restored: epoch {epoch}, batch_size {batch_size}, "
          f"world_size {world_size}, lr {lr}")

    return (
        classifier,
        opt,
        scaler,
        epoch,
        batch_size,
        world_size,
        lr,
    )

def ddp_setup_original(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12346"
    dist.init_process_group(backend="nccl", rank=rank, world_size=world_size)
