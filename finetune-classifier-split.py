"""File to fine tune the classifier.
"""

import yaml
import argparse
import torch
import torch.multiprocessing as mp
from torch.utils.tensorboard import SummaryWriter
from torch.nn.parallel import DistributedDataParallel
import torch.nn.functional as F
import sys
import os
import copy

sys.path.append('/sc/arion/projects/ECHO_ML/gulamf01/multimodal_LLM') # mmllm
sys.path.append('/sc/arion/projects/ECHO_ML/gulamf01/waveform-dataloaders') # wfdm
sys.path.append('/sc/arion/projects/ECHO_ML/gulamf01/jepa') # app AND src
os.environ['CUDA_LAUNCH_BLOCKING']="1" # 
os.environ['TORCH_USE_CUDA_DSA'] = "1" # 

# from mmllm.models.utils import ddp_setup_original
from mmllm.dataloader.video import video_sampler, video_dataloader, video_transforms, video_collator
from mmllm.dataloader.finetune import ClipAggregation, make_transforms
from mmllm.models.classifier import AttentiveClassifier
from wfdl.angiography.dataloader import VideoDataset
from mmllm.models.utils import WarmupCosineSchedule, CosineWDSchedule, pretrain_optimizer, ddp_setup_original
from mmllm.utils import AllGather, AllReduce
from mmllm.models.utils import save_checkpoint, load_checkpoint_finetune
from mmllm.models.viztransformer import vit_large, vit_predictor
from src.models.utils.multimask import MultiMaskWrapper, PredictorMultiMaskWrapper
from src.masks.utils import apply_masks
import pandas as pd # lol
import wandb
from sklearn.metrics import roc_auc_score, average_precision_score
import numpy as np
from sklearn.model_selection import GroupKFold

wandb.init(
    project="cath",
    name="INTERVENTION"
)

torch.cuda.empty_cache()
print("torch.cuda.is_available()", torch.cuda.is_available())  
print("torch.cuda.device_count()", torch.cuda.device_count())  
print("torch.cuda.current_device()",torch.cuda.current_device())

def main(rank, config_kwargs):

    # Load config directory. 
    config_path = config_kwargs.config_path

    # My dictionary. 
    print("Loading dictionary.")
    with open(config_path) as f:
        config = yaml.safe_load(f)
    
    
    # Model metadata. 
    start_epoch = config['start_epoch'] # NEW!
    checkpoint_key = config['checkpoint_key']
    model_name = config['model_name']
    patch_size = config['patch_size']
    pretrain_folder = config['pretrain_folder']
    checkpoint_name = config['ckp_fname']
    tag = config['tag']
    pretrained_path = os.path.join(pretrain_folder, checkpoint_name)

    # Model parameters. 
    use_sdpa = config['use_sdpa']
    use_SiLU = config['use_silu']
    tight_SiLU = config['tight_silu']
    uniform_power = config['uniform_power']
    tubelet_size = config['tubelet_size']
    pretrain_frames_per_clip = config['frames_per_clip']

    # Data path
    train_data_path = config['train_data_path']
    val_data_path = config['val_data_path']
    dataset_type = config['dataset_type']
    num_classes = config['num_classes']
    eval_num_segments = config['num_segments']
    eval_frames_per_clip = config['frames_per_clip']
    eval_frame_step = config['frame_step']
    eval_clip_duration = config['clip_duration']
    eval_num_views_per_segment = config['num_views_per_segment']

    # Optimization
    resolution = config['resolution']
    batch_size = config['batch_size']
    attend_across_segments = config['attend_across_segments']
    num_epochs = config['num_epochs']
    weight_decay = config['weight_decay']
    start_lr = config['start_learning_rate']
    ref_lr = config['learning_rate']
    final_lr = config['final_learning_rate']
    warmup = config['warmup']
    use_bfloat16 = config['use_bfloat16']

    # Troubleshoot
    # device = torch.device("cpu")

    # Setup DDP
    ranks = config['gpus']
    world_size = config['world_size']
    ddp_setup_original(rank=rank, world_size=world_size)
    print("Distributed initialized", torch.distributed.is_initialized())
    # if torch.distributed.is_initialized():
        # gpu_id = torch.distributed.get_rank()
        # torch.cuda.set_device(gpu_id)
        # print("Current gpu", gpu_id)
    
    if torch.cuda.is_available():
        print("Torch is available")
        if torch.distributed.is_initialized():
            print("Torch distributed is initialized")
            gpu_id = torch.distributed.get_rank()
            print("Rank", rank, "GPU-ID", gpu_id)
            # torch.cuda.set_device(gpu_id)
            # print(f"Using GPU {gpu_id}") # Why can't this print??!
            
            # Ensure there are enough GPUs
            num_gpus = torch.cuda.device_count()
            if gpu_id >= num_gpus:
                raise RuntimeError(f"Rank {gpu_id} has an invalid GPU ID. There are only {num_gpus} GPUs.")
            print(f"Rank {rank} will use GPU {gpu_id} out of {num_gpus} available GPUs.")

            try:
                torch.cuda.set_device(gpu_id)
                print(f"Using GPU {gpu_id}")
                device = torch.device(f'cuda:{gpu_id}')  # Define the device
                print("Device has been set")
            except RuntimeError as e:
                print(f"Failed to set GPU {gpu_id}: {e}")
    else:
        print("No GPU available. Using CPU.")
        torch.device("cpu")

    # Logger
    # if(gpu_id == 0):
    writer = SummaryWriter(log_dir=config['log_dir'])

    # Start loading models. 
    ### Load the encoder. 

    #  Model initilization. 
    print("Model initialization")
    uniform_power=config['uniform_power']
    use_mask_tokens=config['use_mask_tokens']
    num_mask_tokens=config['num_mask_tokens']
    zero_init_mask_tokens=config['zero_init_mask_tokens']
    len_cfg_masks=config['len_cfg_masks']

    patch_size=config['patch_size']
    num_frames=config['num_frames']
    # tubelet_size=config['tubelet_size']
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

    # Optimizers. # NEW!
    warmup=config['warmup']
    iterations_per_epoch=config['iterations_per_epoch']
    start_lr= config['start_lr']
    ref_lr= config['ref_lr']
    ipe_scale= config['ipe_scale']
    final_lr= config['final_lr']
    num_epochs=config['num_epochs']
    # mixed_precision=config['mixed_precision']
    weight_decay=config['weight_decay']
    final_weight_decay=config['final_weight_decay']
    loss_exp= config['loss_exp']
    reg_coeff=config['reg_coeff']
    ema=config['ema']
    clip_grad = None

    # Checkpointing # NEW!
    checkpoint_freq = config['checkpoint_frequency']
    checkpoint_folder = config['checkpoint_folder']
    load_model = config['load_model']
    # if load_model:
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
        # load_file = os.listdir(checkpoint_folder)
        # sorted_filenames = sorted(load_file, key=lambda x: int(x.split('-')[0]))
        # load_path = os.path.join(checkpoint_folder, sorted_filenames[-1])
        # print("Loading from file", load_path)
    '''(
        encoder,
        predictor,
        target_encoder,
        optimizer,
        scaler,
        start_epoch,
    ) = load_checkpoint_finetune(
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
    '''

    def _load_snapshot(snapshot_path):
        # snapshot = torch.load(snapshot_path) # original
        snapshot = torch.load(snapshot_path, map_location=torch.device(gpu_id))
        rank = (snapshot['RANK']) # NEW
        encoder = (snapshot['ENCODER'])
        predictor = (snapshot['PREDICTOR'])
        optimizer.load_state_dict(snapshot['OPTIMIZER'])
        # scaler.load_state_dict(snapshot['SCALER'])
        target_encoder = (snapshot['TARGET_ENCODER'])
        start_epoch = snapshot['EPOCH']
        batch_size = snapshot['BATCH_SIZE'] # NEW
        world_size = snapshot['WORLD_SIZE'] # NEW
        lr = snapshot['LEARNING_RATE'] # NEW
    
    snapshot_path = config['snapshot_path']
    if(snapshot_path):
        _load_snapshot(snapshot_path)
    
    ## Add the ClipAggregation
    encoder = ClipAggregation(
            encoder,
            tubelet_size=tubelet_size,
            attend_across_segments=attend_across_segments
    ).to(gpu_id)

    ### Set to evaluation mode. 
    # encoder.eval() # NO! 

    ### Initialize classifier. 
    classifier = AttentiveClassifier(
        embed_dim=encoder.embed_dim,
        num_heads=encoder.num_heads,
        depth=1,
        num_classes=num_classes,
    ).to(gpu_id)

    ### Make data loader
    training = config['training']
    num_views_per_segment = config['num_views_per_segment']
    random_horizontal_flip = config['random_horizontal_flip']
    random_resize_aspect_ratio = config['random_resize_aspect_ratio']
    random_resize_scale = config['random_resize_scale']
    reprob = config['reprob']
    auto_augment = config['auto_augment']
    motion_shift = config['motion_shift']
    crop_size = config['resolution']

    transform = make_transforms(
        training=True,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    transform_for_test = make_transforms(
        training=False,
        num_views_per_clip=num_views_per_segment,
        random_horizontal_flip=random_horizontal_flip,
        random_resize_aspect_ratio=random_resize_aspect_ratio,
        random_resize_scale=random_resize_scale,
        reprob=reprob,
        auto_augment=auto_augment,
        motion_shift=motion_shift,
        crop_size=crop_size,
    )

    # Initialize Video Dataset
    train_data_path = config['train_data_path']
    train_metadata = pd.read_csv(train_data_path)
    train_data_paths = list(train_metadata['path'])
    train_label_paths = list(train_metadata[config['label_name']])

    val_data_path = config['val_data_path']
    val_metadata = pd.read_csv(val_data_path)
    val_data_paths = list(val_metadata['path'])
    val_label_paths = list(val_metadata[config['label_name']])

    # Other dataset info
    datasets_weights = config['dataset_weights']
    frame_step = config['frame_step']
    num_clips = config['num_clips']
    random_clip_sampling = config['random_clip_sampling']
    allow_clip_overlap = config['allow_clip_overlap']
    filter_short_videos = config['filter_short_videos']
    filter_long_videos = config['filter_long_videos']
    shared_transform = config['shared_transform']

    # NEW!
    frames_per_clip = config['frames_per_clip']
    # duration=config['duration']

    traindataset = VideoDataset(
        data_paths=train_data_paths,
        label_paths=train_label_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=None, # NEW! eval_clip_duration, duration. lol why don't these work when their values are legit None.
        shared_transform=shared_transform,
        transform=transform)
    
    valdataset = VideoDataset(
        data_paths=val_data_paths,
        label_paths=val_label_paths,
        datasets_weights=datasets_weights,
        frames_per_clip=frames_per_clip,
        frame_step=frame_step,
        num_clips=num_clips,
        random_clip_sampling=random_clip_sampling,
        allow_clip_overlap=allow_clip_overlap,
        filter_short_videos=filter_short_videos,
        filter_long_videos=filter_long_videos,
        duration=None, # NEW! eval_clip_duration, duration. lol why don't these work when their values are legit None.
        shared_transform=shared_transform,
        transform=transform_for_test)

    # Initialize supervised sampler. 
    shuffle = config['shuffle']
    # world_size = config['world_size'] # mentioned up
    rank = gpu_id
    trainsupervised_sampler = video_sampler(
        traindataset, world_size, rank, shuffle
    )
    valsupervised_sampler = video_sampler(
        valdataset, world_size, rank, shuffle
    )

    # Video Dataloader
    batch_size = config['batch_size']
    drop_last = config['drop_last']
    num_workers = config['num_workers']
    pin_memory = config['pin_memory']
    train_loader = video_dataloader(
        traindataset, 
        collator=None,
        sampler=trainsupervised_sampler, 
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    val_loader = video_dataloader(
        valdataset, 
        collator=None,
        sampler=valsupervised_sampler, 
        batch_size=batch_size,
        drop_last=drop_last,
        num_workers=num_workers,
        pin_memory=pin_memory
    )
    '''
    train_loader = iter(supervised_loader)
    # Use iter() when (1) need explicit control over the iterator (e.g., mixing multiple iterators).
    # (2) implement custom iteration logic.
    # Otherwise, directly looping over dataLoader is more straightforward/ avoids confusion
    '''
    ## Initailize optimizer. 
    param_groups = [
        {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' not in n) and (len(p.shape) != 1))
        }, {
            'params': (p for n, p in classifier.named_parameters()
                       if ('bias' in n) or (len(p.shape) == 1)),
            'WD_exclude': True,
            'weight_decay': 0
        }
    ]
    optimizer = torch.optim.AdamW(param_groups)

    # Initialize scheduler
    iterations_per_epoch = len(train_loader)
    final_weight_decay = config['final_weight_decay']
    
    learning_rate_scheduler = WarmupCosineSchedule(
        optimizer,
        warmup_steps=int(warmup*iterations_per_epoch),
        start_lr=start_lr,
        ref_lr=ref_lr,
        final_lr=final_lr,
        T_max=int(num_epochs*iterations_per_epoch))
    weight_decay_scheduler = CosineWDSchedule(
        optimizer,
        ref_wd=weight_decay,
        final_wd=final_weight_decay, # NEW!
        T_max=int(num_epochs*iterations_per_epoch))
    scaler = torch.cuda.amp.GradScaler() if use_bfloat16 else None

    # Initialize computation graph. 
    classifier = DistributedDataParallel(classifier, static_graph=True)

    checkpoint_folder = config['checkpoint_folder']
    load_model = config['load_model']

    if load_model and len(os.listdir(checkpoint_folder)) > 0:
        # Don't forget to change checkpoint_folder!!
        print("Loading from checkpoint.")
        load_file = os.listdir(checkpoint_folder)
        sorted_filenames = sorted(load_file, key=lambda x: int(x.split('-')[0]))
        load_path = os.path.join(checkpoint_folder, sorted_filenames[-1])
        latest_file = f'{epoch}-latest-pth.tar'
        # latest_file = '1-latest-pth.tar'
        latest_path = os.path.join(checkpoint_folder, latest_file)
        '''
        classifier, optimizer, scaler, start_epoch = load_checkpoint(
            device=device,
            r_path=latest_path,
            classifier=classifier,
            opt=optimizer,
            scaler=scaler)
            '''
        (
            classifier,
            optimizer,
            scaler,
            epoch,
            batch_size,
            world_size,
            lr,
        ) = load_checkpoint_finetune(
            checkpoint_path=latest_path, 
            classifier=classifier, 
            opt=optimizer, 
            scaler=scaler)
        for _ in range(start_epoch*ipe_scale):
            scheduler.step()
            wd_scheduler.step()


    def save_checkpoint(epoch):
        save_dict = {
            'classifier': classifier.state_dict(),
            'opt': optimizer.state_dict(),
            'scaler': None if scaler is None else scaler.state_dict(),
            'epoch': epoch,
            'batch_size': batch_size,
            'world_size': world_size,
            'lr': start_lr
        }
        latest_file = f'{epoch}-latest-pth.tar'
        latest_path = os.path.join(checkpoint_folder, latest_file)
        if rank == 0:
            torch.save(save_dict, latest_path)

    # Start training
    ## Initialize training loop
    classifier.train(mode=training)
    criterion = torch.nn.CrossEntropyLoss()

    # https://stackoverflow.com/questions/74805555/why-does-my-training-loop-completely-skip-after-the-first-epoch-in-pytorch
    # Not CUDA Version Mismatch, Driver Issues, Improper Tensor Device Placement, Corrupted PyTorch Installation, Multi-GPU or Distributed Training
    # Could it be GPU Memory Exhaustion, Data Corruption, Code Logic Issues i.e. bug in how tensors or models are handled might surface only after enough iterations (e.g., model weights growing abnormally large)
    torch.cuda.empty_cache()
    # torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    torch.cuda.synchronize()
    torch.backends.cudnn.benchmark = True
    for epoch in range(start_epoch, num_epochs):
        all_outputs = []
        all_labels = []
        correct_predictions = 0
        total_samples = 0
        epoch_loss = 0.0
        num_batches = 0
        print(f"New epoch starting now: {epoch}")

        if train_loader is None:
            print("Issue with train_loader!")

        # Training Loop
        for itr in range(iterations_per_epoch):
        # for itr, data in enumerate(train_loader):
            # if itr >= 10: break  # Stop after the first 10 batches
            data = next(iter(train_loader), None)  # Get the next batch or None if exhausted
            if itr >= 3000: # if data is None:
                break

            if itr % 1000 == 0:
                print(f"We are still processing, iteration {itr} now")
            
            # print(f"Iteration {itr}: Allocated {torch.cuda.memory_allocated()} bytes; Memory Usage {torch.cuda.memory_reserved()} bytes")
            
            # Scheduler updates
            learning_rate_scheduler.step()
            weight_decay_scheduler.step()

            # Prepare inputs and labels
            clips = [[dij.to(device) for dij in di] for di in data[0]]
            clip_indices = [d.to(device) for d in data[2]]
            labels = data[1].long().to(device)

            # Forward pass
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
                raw_outputs = encoder(clips, clip_indices)

                # Classifier outputs
                if attend_across_segments:
                    logits = [classifier(o).float() for o in raw_outputs]
                else:
                    logits = [[classifier(ost).float() for ost in os] for os in raw_outputs]

                # Compute loss
                if training:
                    if attend_across_segments:
                        loss = sum([criterion(logit, labels) for logit in logits]) / len(logits)
                    else:
                        loss = sum([sum([criterion(ost, labels) for ost in os]) for os in logits]) / len(logits) / len(logits[0])

                # Post-process logits for evaluation
                with torch.no_grad():
                    if attend_across_segments:
                        probabilities = sum([F.softmax(logit, dim=1) for logit in logits]) / len(logits)
                    else:
                        probabilities = sum([sum([F.softmax(ost, dim=1) for ost in os]) for os in logits]) / len(logits) / len(logits[0])

                # Calculate accuracy
                correct_predictions += probabilities.max(dim=1).indices.eq(labels).sum().item()
                total_samples += labels.size(0)

            # Use logits for loss, probabilities for metrics
            if probabilities is not None and labels is not None:
                all_outputs.extend(probabilities.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())
            else:
                print(f"Skipping batch {itr} due to missing probabilities or labels.")

            epoch_loss += loss.item()
            num_batches += 1 

            # Back-propagate and update scaler and optimizer. 
            if training:
                if use_bfloat16:
                    # Scale.
                    scaler.scale(loss).backward()
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                    scaler.step(optimizer)
                    scaler.update()
                else:
                    # Back prop
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                    optimizer.step()
                optimizer.zero_grad()
                memory = torch.cuda.max_memory_allocated() / 1024.**2

                wandb.log({'Iterations': itr})

        # After each epoch, compute AUROC and AUPRC.
        if len(all_outputs) == 0:
            print("Empty outputs!")

        if len(all_labels) == 0:
            print("Empty labels!")

        if all_outputs:
            epoch_outputs = np.array([output[1] for output in all_outputs])
        else:
            print(f"Missing all_outputs: {len(all_outputs)} ")
            epoch_outputs = np.array([])

        if all_labels:
            epoch_labels = np.array(all_labels)
        else:
            print(f"Missing all_labels: {len(all_labels)} ")
            epoch_labels = np.array([])

        # AUROC, AUPRC for training set
        auroc = roc_auc_score(epoch_labels, epoch_outputs)
        auprc = average_precision_score(epoch_labels, epoch_outputs)

        wandb.log({"Epoch": epoch+1, "AUROC (Train)": auroc, "AUPRC (Train)": auprc})
        epoch_accuracy = 100. * correct_predictions / total_samples
        wandb.log({"Top-1 Accuracy (Train)": epoch_accuracy})
        epoch_loss /= num_batches
        print(f"Epoch {epoch+1}/{num_epochs}, Loss (Train): {epoch_loss:.4f}")
        wandb.log({"Loss (Train)": epoch_loss})

        # Validation Evaluation
        val_outputs = []
        val_labels = []
        val_correct_predictions = 0
        val_total_samples = 0
        val_loss = 0.0
        encoder.eval() 
        with torch.no_grad():
            # for val_itr, val_data in enumerate(val_loader):
            for itr in range(iterations_per_epoch):

                if itr >= 3000: # if data is None:
                    break

                val_data = next(iter(val_loader), None)
                clips = [[dij.to(device) for dij in di] for di in val_data[0]]
                clip_indices = [d.to(device) for d in val_data[2]]
                val_labels_batch = val_data[1].to(device)

                # Forward pass
                raw_val_outputs = encoder(clips, clip_indices)
                if attend_across_segments:
                    val_logits = [classifier(o) for o in raw_val_outputs]
                else:
                    val_logits = [[classifier(ost) for ost in os] for os in raw_val_outputs]

                # Compute loss
                if attend_across_segments:
                    val_loss_batch = sum([criterion(logit, val_labels_batch) for logit in val_logits]) / len(val_logits)
                else:
                    val_loss_batch = sum([sum([criterion(ost, val_labels_batch) for ost in os]) for os in val_logits]) / len(val_logits) / len(val_logits[0])

                val_loss += val_loss_batch.item()
                # Post-process logits for evaluation
                if attend_across_segments:
                    val_probabilities = sum([F.softmax(logit, dim=1) for logit in val_logits]) / len(val_logits)
                else:
                    val_probabilities = sum([sum([F.softmax(ost, dim=1) for ost in os]) for os in val_logits]) / len(val_logits) / len(val_logits[0])

                val_correct_predictions += val_probabilities.max(dim=1).indices.eq(val_labels_batch).sum().item()
                val_total_samples += val_labels_batch.size(0)

                # Collect outputs for metrics
                val_outputs.extend(val_probabilities.cpu().numpy())
                val_labels.extend(val_labels_batch.cpu().numpy())

        val_loss /= len(val_loader)
        val_accuracy = 100. * val_correct_predictions / val_total_samples
        val_auroc = roc_auc_score(val_labels, val_outputs)
        val_auprc = average_precision_score(val_labels, val_outputs)

        # Log validation metrics
        wandb.log({
            "Epoch": epoch + 1,
            "Loss (Validation)": val_loss,
            "Top-1 Accuracy (Validation)": val_accuracy,
            "AUROC (Validation)": val_auroc,
            "AUPRC (Validation)": val_auprc
        })

        print(f"Epoch {epoch+1}/{num_epochs}, Loss (Validation): {val_loss:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, Accuracy (Validation): {val_accuracy:.2f}%")
        print(f"Epoch {epoch+1}/{num_epochs}, AUROC (Validation): {val_auroc:.4f}")
        print(f"Epoch {epoch+1}/{num_epochs}, AUPRC (Validation): {val_auprc:.4f}")

        # Save checkpoint
        # save_checkpoint(epoch)
        print("Completed an epoch!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Configuration arguments for fine-tuning.')
    
    parser.add_argument("--config_path", required=True, type=str,
                        help="Config file path.")

    config_kwargs = parser.parse_args()

    # Spawn the world size. 
    world_size = 3
    mp.spawn(main, args=(config_kwargs,), nprocs=world_size)

    '''
    for epoch in range(start_epoch, num_epochs):
        all_outputs = []
        all_labels = []
        correct_predictions = 0
        total_samples = 0
        for itr, data in enumerate(train_loader):

            if len(data) == 0:
                raise ValueError("Empty batch encountered. Check your DataLoader.")
    
            # Step with the schedulers. 
            learning_rate_scheduler.step()
            weight_decay_scheduler.step()

            # Cast to bfloat
            with torch.cuda.amp.autocast(dtype=torch.float16, enabled=use_bfloat16):
                
                # Iterate over the spatial and temporal view of the clip
                clips = [
                    [dij.to(device, non_blocking=True) for dij in di]  # iterate over spatial views of clip
                    for di in data[0]  # iterate over temporal index of clip
                ]
                clip_indices = [d.to(device, non_blocking=True) for d in data[2]]
                
                # Labels
                labels = data[1].to(device)

                # Batch size. 
                batch_size = len(labels)

                # Torch no grad. 
                with torch.no_grad():
                    outputs = encoder(clips, clip_indices)
                    # print("outputs shape", outputs[0].shape)
                    if outputs is None or len(outputs) == 0:
                        raise ValueError("Model produced no output. Check your data or model logic.")

                    if not training:
                        if attend_across_segments:
                            outputs = [classifier(o) for o in outputs]
                        else:
                            outputs = [[classifier(ost) for ost in os] for os in outputs]
                
                if training:
                    if attend_across_segments:
                        outputs = [classifier(o) for o in outputs]
                    else:
                        outputs = [[classifier(ost) for ost in os] for os in outputs]
                
                # Compute loss
                if attend_across_segments:
                    loss = sum([criterion(o, labels) for o in outputs]) / len(outputs)
                else:
                    loss = sum([sum([criterion(ost, labels) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0])
                
                with torch.no_grad():
                    if attend_across_segments:
                        outputs = sum([F.softmax(o, dim=1) for o in outputs]) / len(outputs)
                    else:
                        outputs = sum([sum([F.softmax(ost, dim=1) for ost in os]) for os in outputs]) / len(outputs) / len(outputs[0])# Store predictions and labels for metrics
                   
                   # Store predictions and labels for metrics
                    all_outputs.append(outputs.cpu().numpy())
                    all_labels.append(labels.cpu().numpy())
                    
                    # Calculate accuracy
                    # top1_acc = 100. * outputs.max(dim=1).indices.eq(labels).sum() / batch_size
                    # top1_acc = float(AllReduce.apply(top1_acc))
                    # wandb.log({'Accuracy': top1_acc})

                    # Top-1 Accuracy for the current batch
                    correct_predictions += outputs.max(dim=1).indices.eq(labels).sum().item()
                    total_samples += labels.size(0)
                
                # Back-propagate and update scaler and optimizer. 
                if training:
                    if use_bfloat16:
                        # Scale.
                        scaler.scale(loss).backward()
                        scaler.unscale_(optimizer)
                        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                        scaler.step(optimizer)
                        scaler.update()
                    else:
                        # Back prop
                        loss.backward()
                        torch.nn.utils.clip_grad_norm_(classifier.parameters(), 1.0)
                        optimizer.step()
                    optimizer.zero_grad()
                    memory = torch.cuda.max_memory_allocated() / 1024.**2

                    # wandb.log({'Iterations': itr})
                    # wandb.log({'Loss': loss})
                    # wandb.log({'Memory': memory})
                '''
