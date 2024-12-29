
import torch
import math

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
    scaler):
    checkpoint = torch.load(r_path, map_location=torch.device('cpu'))
    epoch = checkpoint['epoch']

    # -- loading encoder
    pretrained_dict = checkpoint['encoder']
    msg = encoder.load_state_dict(pretrained_dict)
    #logger.info(f'loaded pretrained encoder from epoch {epoch} with msg: {msg}')

    # -- loading predictor
    pretrained_dict = checkpoint['predictor']
    msg = predictor.load_state_dict(pretrained_dict)

    # -- loading optimizer
    opt.load_state_dict(checkpoint['opt'])
    if scaler is not None:
        scaler.load_state_dict(checkpoint['scaler'])

    # -- loading target_encoder
    if target_encoder is not None:
        print(list(checkpoint.keys()))
        pretrained_dict = checkpoint['target_encoder']
        msg = target_encoder.load_state_dict(pretrained_dict)
    
    return (
        encoder,
        predictor,
        target_encoder,
        opt,
        scaler,
        epoch,
    )
