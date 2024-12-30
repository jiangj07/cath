import argparse
import os
import yaml
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import copy
import torch.nn.functional as F
from torch.utils.tensorboard import SummaryWriter
from torch.distributed import init_process_group
import torch.multiprocessing as mp
from torch.utils.data.distributed import DistributedSampler
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.distributed import init_process_group, destroy_process_group
import sys

sys.path.append('/sc/arion/projects/ECHO_ML/gulamf01/multimodal_LLM') # mmllm
sys.path.append('/sc/arion/projects/ECHO_ML/gulamf01/waveform-dataloaders') # wfdm
sys.path.append('/sc/arion/projects/ECHO_ML/gulamf01/jepa') # app AND src
#os.environ['CUDA_LAUNCH_BLOCKING']="1"#
# os.environ['TORCH_USE_CUDA_DSA'] = "1"

from mmllm.dataloader.video import video_sampler, video_dataloader, video_transforms, video_collator
from wfdl.angiography.dataloader import VideoDataset
from mmllm.dataloader.utils import repeat_interleave_batch
from mmllm.models.utils import save_checkpoint, load_checkpoint
from mmllm.models.viztransformer import vit_large, vit_predictor
from mmllm.models.utils import WarmupCosineSchedule, CosineWDSchedule, pretrain_optimizer
from app.vjepa.utils import init_video_model
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.masks.utils import apply_masks






def main(rank, config_kwargs):
    
    # rank = int(os.environ["LOCAL_RANK"])
    config_path = config_kwargs.config_path
    # My dictionary. 
    print("Loading dictionary.")
    with open(config_path) as f:
        config = yaml.safe_load(f)

    print("Loading Logger")
    writer = SummaryWriter(log_dir=config['log_dir'])

    start_epoch = config['start_epoch']
    # epoch=start_epoch
    num_epochs = config['num_epochs']
    mask_type = config['mask_type']
    data_path = config['data_path']
    metadata = pd.read_table(data_path)
    data_paths = list(metadata['path'])
    datasets_weights=config['datasets_weights']
    frames_per_clip = config['frames_per_clip']
    frame_step= config['frame_step'] # sampling rate
    num_clips = config['num_clips']
    random_clip_sampling=config['random_clip_sampling']
    allow_clip_overlap=config['allow_clip_overlap']
    filter_short_videos=config['filter_short_videos']
    filter_long_videos=config['filter_long_videos']
    duration=config['duration']
    shared_transform=config['shared_transform']
    snapshot_path=config['snapshot_path']

    ranks = config['gpus']
    world_size=config['world_size']
    shuffle=config['shuffle']
    ddp_setup_original(rank=rank, world_size=world_size)

    print("Distributed initizlied", torch.distributed.is_initialized())
    if torch.distributed.is_initialized():
        gpu_id = torch.distributed.get_rank()
        torch.cuda.set_device(gpu_id)
        print("Current gpu", gpu_id)


    # if not torch.cuda.is_available():
    #     device = torch.device('cpu')
    # else:
    #     device = torch.device('cuda')
    #     torch.cuda.set_device(device)

    # Samplers
    
    # rank=config['rank']
    

    

    # Loaders
    batch_size=config['batch_size']
    drop_last=config['drop_last']
    pin_memory=config['pin_memory']
    num_workers=config['num_workers']
    seed = config['seed']    # seed everything

    print("Loading transforms.")
    np.random.seed(config['seed'])
    torch.manual_seed(seed)
    transform = video_transforms(
        random_horizontal_flip=config['random_horizontal_flip'],
        random_resize_aspect_ratio=config['random_resize_aspect_ratio'],
        random_resize_scale=config['random_resize_scale'],
        reprob=config['reprob'],
        auto_augment=config['auto_augment'],
        motion_shift=config['motion_shift'],
        crop_size=config['crop_size']
    )
    collator = video_collator(
        crop_size=config['crop_size'],
        num_frames=config['num_frames'],
        patch_size=config['patch_size'],
        tubelet_size=config['tubelet_size'],
        cfgs_mask=config['mask']
    )

    print("Loading dataset.")
    dataset = VideoDataset(
            data_paths=data_paths,
            datasets_weights=None,
            frames_per_clip=frames_per_clip,
            frame_step=frame_step,
            num_clips=num_clips,
            random_clip_sampling=random_clip_sampling,
            allow_clip_overlap=allow_clip_overlap,
            filter_short_videos=filter_short_videos,
            filter_long_videos=filter_long_videos,
            duration=None,
            shared_transform=None,
            transform=transform
    )

    print("loading sampler.")
    unsupervised_sampler = video_sampler(
        dataset, world_size, rank, shuffle
    )

    print("Loading loader.")
    unsupervised_loader = video_dataloader(
        dataset, 
        collator=collator,
        sampler=unsupervised_sampler, 
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    loader = iter(unsupervised_loader)

    #  Model initilization. 
    print("Model initialization")
    uniform_power=config['uniform_power']
    use_mask_tokens=config['use_mask_tokens']
    num_mask_tokens=config['num_mask_tokens']
    zero_init_mask_tokens=config['zero_init_mask_tokens']
    len_cfg_masks=config['len_cfg_masks']

    patch_size=config['patch_size']
    num_frames=config['num_frames']
    tubelet_size=config['tubelet_size']
    model_name=config['model_name']
    crop_size=config['crop_size']
    pred_depth=config['pred_depth']
    pred_embed_dim=config['pred_embed_dim']
    use_sdpa=config['use_sdpa']

    # Initialize model.
    print("Initializing model.")
    encoder = vit_large(
            img_size=crop_size,
            patch_size=patch_size,
            num_frames=num_frames,
            tubelet_size=tubelet_size,
            uniform_power=uniform_power,
            use_sdpa=use_sdpa,
    )

    print("Initializing predictor.")
    encoder = MultiMaskWrapper(encoder)
    predictor = vit_predictor(
        img_size=crop_size,
        use_mask_tokens=use_mask_tokens,
        patch_size=patch_size,
        num_frames=num_frames,
        tubelet_size=tubelet_size,
        embed_dim=encoder.backbone.embed_dim,
        predictor_embed_dim=pred_embed_dim,
        depth=pred_depth,
        num_heads=encoder.backbone.num_heads,
        uniform_power=uniform_power,
        num_mask_tokens=num_mask_tokens,
        zero_init_mask_tokens=zero_init_mask_tokens,
        use_sdpa=use_sdpa
    )
    predictor = PredictorMultiMaskWrapper(predictor)

    target_encoder = copy.deepcopy(encoder)


    zero_init_bias_wd=config['zero_init_bias_wd']
    betas=config['betas']
    epsilon=config['epsilon']
    mixed_precision = config['mixed_precision']

    

    # Optimizers. 
    warmup=config['warmup']
    iterations_per_epoch=config['iterations_per_epoch']
    start_lr= config['start_lr']
    ref_lr= config['ref_lr']
    ipe_scale= config['ipe_scale']
    final_lr= config['final_lr']
    num_epochs=config['num_epochs']
    mixed_precision=config['mixed_precision']
    weight_decay=config['weight_decay']
    final_weight_decay=config['final_weight_decay']
    loss_exp= config['loss_exp']
    reg_coeff=config['reg_coeff']
    ema=config['ema']
    clip_grad = None

    
    
    # Checkpointing
    checkpoint_freq = config['checkpoint_frequency']
    checkpoint_folder = config['checkpoint_folder']
    load_model = config['load_model']
    print("Loading from checkpoint.")
    if load_model:
        print("Initializing optimizer.")
        optimizer = pretrain_optimizer(
            encoder=encoder, predictor=predictor, betas=betas, epsilon=epsilon, zero_init_bias_wd=zero_init_bias_wd
        )
        scheduler = WarmupCosineSchedule(
                optimizer,
                warmup_steps=int(warmup * iterations_per_epoch),
                start_lr=start_lr,
                ref_lr=ref_lr,
                final_lr=final_lr,
                T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )


        wd_scheduler = CosineWDSchedule(
                optimizer,
                ref_wd=weight_decay,
                final_wd=final_weight_decay,
                T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
            )

        momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(iterations_per_epoch*num_epochs*ipe_scale)
                                for i in range(int(iterations_per_epoch*num_epochs*ipe_scale)+1))

        scaler = torch.cuda.amp.GradScaler() if mixed_precision else None
        load_file = os.listdir(checkpoint_folder)
        sorted_filenames = sorted(load_file, key=lambda x: int(x.split('-')[0]))
        load_path = os.path.join(checkpoint_folder, sorted_filenames[-1])
        # load_path = '/sc/arion/projects/ECHO_ML/gulamf01/multimodal_LLM/src/1_pretrain-model/1.2.3_train-loop-gpu-huge/checkpoint/97-latest-pth.tar'
        
        print("Loading from file", load_path)
        (
            encoder,
            predictor,
            target_encoder,
            optimizer,
            scaler,
            start_epoch,
        ) = load_checkpoint( # Problem line!!
            r_path=load_path,
            encoder=encoder,
            predictor=predictor,
            target_encoder=target_encoder,
            opt=optimizer,
            scaler=scaler,
            device=gpu_id)
        print("Current on start epoch", start_epoch)
        for _ in range(start_epoch * iterations_per_epoch):
            scheduler.step()
            wd_scheduler.step()
            next(momentum_scheduler)
            collator.step()
        print(start_epoch)
    
    def _load_snapshot(snapshot_path):
        snapshot = torch.load(snapshot_path)
        encoder = (snapshot['ENCODER'])
        predictor = (snapshot['PREDICTOR'])
        target_encoder = (snapshot['TARGET_ENCODER'])
        optimizer.load_state_dict(snapshot['OPTIMIZER'])
        scaler.load_state_dict(snapshot['SCALER'])
        start_epoch = snapshot['EPOCH']
    
    def _save_snapshot(rank, encoder, predictor, optimizer, scaler, target_encoder, epoch, batch_size, world_size, lr, path):
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
            'RANK': rank
        }
        torch.save(save_dict, snapshot_path)
    #if(os.path.exists(snapshot_path)):
    #    _load_snapshot()
    
    # ddp_setup()
    # print(torch.cuda.memory_summary())
    

    print("Rank", rank, "GPU-ID", gpu_id)
    try:
        encoder = encoder.to(gpu_id)
        encoder = DDP(encoder, device_ids = [gpu_id],find_unused_parameters=True)

        predictor = predictor.to(gpu_id)
        predictor = DDP(predictor, device_ids = [gpu_id],find_unused_parameters=True)

        target_encoder = target_encoder.to(gpu_id)
        target_encoder = DDP(target_encoder, device_ids = [gpu_id],find_unused_parameters=True)
        print("Succesfully loaded onto GPU", gpu_id)
    except:
        print("Failed to load on GPU", gpu_id, "Rank", rank)
    #predictor.to(gpu_id)
    #target_encoder.to(gpu_id)

    print("Initializing optimizer.")
    optimizer = pretrain_optimizer(
        encoder=encoder, predictor=predictor, betas=betas, epsilon=epsilon, zero_init_bias_wd=zero_init_bias_wd
    )
    scaler = torch.cuda.amp.GradScaler() if mixed_precision else None

    print("Initializing scheduler.")
    dtype=torch.bfloat16
    scheduler = WarmupCosineSchedule(
            optimizer,
            warmup_steps=int(warmup * iterations_per_epoch),
            start_lr=start_lr,
            ref_lr=ref_lr,
            final_lr=final_lr,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )


    wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=weight_decay,
            final_wd=final_weight_decay,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
        )

    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(iterations_per_epoch*num_epochs*ipe_scale)
                            for i in range(int(iterations_per_epoch*num_epochs*ipe_scale)+1))


    scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup * iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )


    wd_scheduler = CosineWDSchedule(
            optimizer,
            ref_wd=weight_decay,
            final_wd=final_weight_decay,
            T_max=int(ipe_scale * num_epochs * iterations_per_epoch),
    )

    momentum_scheduler = (ema[0] + i*(ema[1]-ema[0])/(iterations_per_epoch*num_epochs*ipe_scale)
                            for i in range(int(iterations_per_epoch*num_epochs*ipe_scale)+1))


    for epoch in range(start_epoch, num_epochs):
        unsupervised_sampler.set_epoch(epoch)

        print("Currently on epoch", epoch)

        # Iterations
        #itr = 1
        #ipe = 1

        latent_list = []
        images_list = []
        for itr in range(iterations_per_epoch):
            # print("Current iteration", current_itr)
            current_itr = int(itr + epoch*iterations_per_epoch)
            '''
            udata, masks_enc, masks_pred = next(loader)
            print("On iteration", itr)
            '''
            try:
                udata, masks_enc, masks_pred = next(loader) # labels, clip_indices!
            except StopIteration:
                loader = iter(unsupervised_loader)
                udata, masks_enc, masks_pred = next(loader)

            def load_clips():
                # -- unsupervised video clips
                # Put each clip on the GPU and concatenate along batch
                # dimension
                clips = torch.cat([u.to(gpu_id, non_blocking=True) for u in udata[0]], dim=0)

                # Put each mask-enc/mask-pred pair on the GPU and reuse the
                # same mask pair for each clip
                _masks_enc, _masks_pred = [], []
                for _me, _mp in zip(masks_enc, masks_pred):
                    _me = _me.to(gpu_id, non_blocking=True)
                    _mp = _mp.to(gpu_id, non_blocking=True)
                    _me = repeat_interleave_batch(_me, batch_size, repeat=num_clips)
                    _mp = repeat_interleave_batch(_mp, batch_size, repeat=num_clips)
                    _masks_enc.append(_me)
                    _masks_pred.append(_mp)

                return (clips, _masks_enc, _masks_pred)
            clips, masks_enc, masks_pred = load_clips()

            # print('clips', clips.shape, 'mask', len(masks_enc), 'masks_preds', len(masks_pred))

            def train_step():
                _new_lr = scheduler.step()
                _new_wd = wd_scheduler.step()
            
                def forward_target(c):
                    """
                    Returns list of tensors of shape [B, N, D], one for each
                    mask-pred.
                    """
                    with torch.no_grad():
                        h = target_encoder(c)
                        h = F.layer_norm(h, (h.size(-1),))  # normalize over feature-dim  [B, N, D]
                        # -- create targets (masked regions of h)
                        h = apply_masks(h, masks_pred, concat=False)
                        return h

                def forward_context(c, h):
                    """
                    Returns list of tensors of shape [B, N, D], one for each
                    mask-pred.
                    """
                    z = encoder(c, masks_enc)
                    z = predictor(z, h, masks_enc, masks_pred)
                    return z

                def loss_fn(z, h):
                    loss = 0.
                    # Compute loss and accumulate for each mask-enc/mask-pred pair
                    for zi, hi in zip(z, h):
                        loss += torch.mean(torch.abs(zi - hi)**loss_exp) / loss_exp
                    loss /= len(masks_pred)
                    return loss

                def reg_fn(z):
                    return sum([torch.sqrt(zi.var(dim=1) + 0.0001) for zi in z]) / len(z)
                
                # Step 1. Forward
                loss_jepa, loss_reg = 0., 0.
                #with torch.cuda.amp.autocast(dtype=dtype, enabled=mixed_precision):
                h = forward_target(clips)
                z = forward_context(clips, h)
                loss_jepa = loss_fn(z, h)  # jepa prediction loss
                pstd_z = reg_fn(z)  # predictor variance across patches
                loss_reg += torch.mean(F.relu(1.-pstd_z))
                loss = loss_jepa + reg_coeff * loss_reg
                if(rank == 0):
                    writer.add_scalar("Loss/JEPA", loss_jepa, current_itr)
                    # writer.add_scalar("Loss/PSTD", pstd_z, current_itr)
                    writer.add_scalar("Loss/Regularized", loss_reg, current_itr)
                    writer.add_scalar("Loss/total", loss, current_itr)

                # Step 2. Backward & step
                _enc_norm, _pred_norm = 0., 0.
                if mixed_precision:
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                else:
                    loss.backward()
                if (epoch > warmup) and (clip_grad is not None):
                    _enc_norm = torch.nn.utils.clip_grad_norm_(encoder.parameters(), clip_grad)
                    _pred_norm = torch.nn.utils.clip_grad_norm_(predictor.parameters(), clip_grad)
                if mixed_precision:
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    optimizer.step()
                
                optimizer.zero_grad()

                # Step 3. momentum update of target encoder
                m = next(momentum_scheduler)
                with torch.no_grad():
                    """_summary_

                    Args:
                        loss_reg (_type_, optional): _description_. Defaults to 0..
                        _pred_norm (_type_, optional): _description_. Defaults to 0..
                    """                    
                    for param_q, param_k in zip(encoder.parameters(), target_encoder.parameters()):
                        param_k.data.mul_(m).add_((1.-m) * param_q.detach().data)
            # print("Running train step.")
            train_step()


        if epoch % checkpoint_freq == 0 or epoch == (num_epochs - 1):
            latest_file = f'{epoch}-latest-pth.tar'
            latest_path = os.path.join(checkpoint_folder, latest_file)
            
            if(world_size == 1):
                save_checkpoint(
                    rank=0, 
                    encoder=encoder,
                    predictor=predictor,
                    optimizer=optimizer,
                    scaler=scaler,
                    target_encoder=target_encoder,
                    epoch=epoch,
                    batch_size=batch_size,
                    world_size=world_size, 
                    lr=start_lr,
                    path= latest_path
                )
            else:
                if(world_size != 1 and rank == 0):
                    save_checkpoint(
                        rank=rank, 
                        encoder=encoder,
                        predictor=predictor,
                        optimizer=optimizer,
                        scaler=scaler,
                        target_encoder=target_encoder,
                        epoch=epoch,
                        batch_size=batch_size,
                        world_size=world_size, 
                        lr=start_lr,
                        path= latest_path
                    )
                    _save_snapshot(
                        rank=rank,
                        encoder=encoder,
                        predictor=predictor,
                        optimizer=optimizer,
                        scaler=scaler,
                        target_encoder=target_encoder,
                        epoch=epoch,
                        batch_size=batch_size,
                        world_size=world_size, 
                        lr=start_lr,
                        path= latest_path
                    )
                        # save_checkpoint(
                        #     rank=0, 
                        #     encoder=encoder.module,
                        #     predictor=predictor.module,
                        #     optimizer=optimizer,
                        #     scaler=scaler,
                        #     target_encoder=target_encoder.module,
                        #     epoch=epoch,
                        #     batch_size=batch_size,
                        #     world_size=world_size, 
                        #     lr=start_lr,
                        #     path= latest_path
    destroy_process_group()
    return None


def ddp_setup():
    init_process_group(backend="nccl")


def ddp_setup_original(rank, world_size):
    os.environ["MASTER_ADDR"] = "localhost"
    os.environ["MASTER_PORT"] = "12346"
    init_process_group(backend="nccl", rank=rank, world_size=world_size)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration arguments for pre-training.')
    
    parser.add_argument("--config_path", required=True, type=str,
                        help="Config file path.")

    config_kwargs = parser.parse_args()

    # Spawn the world size. 
    world_size = 3
    mp.spawn(main, args=(config_kwargs,), nprocs=world_size)

    # With torchrun
    # main(config_kwargs)
