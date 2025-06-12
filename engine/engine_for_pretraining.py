import math
import sys
from typing import Iterable
import torch
import torch.nn as nn
from discovr.utils import utils
from einops import rearrange
from timm.data.constants import IMAGENET_DEFAULT_MEAN, IMAGENET_DEFAULT_STD
import torch.nn.functional as F
from discovr.utils.visualize_masks import denormalize_video, overlay_video_cmap
from PIL import Image
import wandb
from discovr.models.info_nce import InfoNCE, info_nce
from timm.models.vision_transformer import vit_small_patch16_224
from sklearn.metrics import balanced_accuracy_score, f1_score, precision_score, recall_score
import numpy as np
from discovr.utils.utils import chromatic_correction_fct


def train_one_epoch(model: torch.nn.Module, feature_extraction_model:torch.nn.Module, data_loader: Iterable, optimizer: torch.optim.Optimizer,
                    device: torch.device, epoch: int, loss_scaler, max_norm: float = 0, patch_size: int = 16, mask_type='tube',
                    target_type='pixel', normlize_target: bool = True, log_writer=None, lr_scheduler=None, start_steps=None,
                    lr_schedule_values=None, wd_schedule_values=None, output_dir='./', loss_func='L2',
                    swav_weight=1.0, dino_weight=1.0, turbo_weight=1.0, video_dino_weight=1.0, dino_only=False, use_video_dino=False,grayscale_mode=False):
    model.train()
    metric_logger = utils.MetricLogger(delimiter="  ")
    header = 'Epoch: [{}]'.format(epoch)
    print_freq = 10
    
    # Print turbo weight for debugging
    if hasattr(model.module, 'use_turbo_training') and model.module.use_turbo_training:
        print(f"Using Turbo Training with weight: {turbo_weight}")
    
    if loss_func == 'L2':
        loss_cpt = nn.MSELoss()
    elif loss_func == 'SWAV':
        loss_cpt = nn.CrossEntropyLoss()

    # Add gradient checking function without disturbing existing code
    def print_grad_norms(model, phase=""):
        total_norm = 0
        param_norms = {}
        for name, p in model.named_parameters():
            if p.grad is not None:
                param_norm = p.grad.data.norm(2)
                total_norm += param_norm.item() ** 2
                param_norms[name] = param_norm.item()
        total_norm = total_norm ** 0.5
        print(f"\n=== Gradient Norms {phase} ===")
        print(f"Total norm: {total_norm:.3f}")
        print("Top 5 gradient norms:")
        top_grads = sorted(param_norms.items(), key=lambda x: x[1], reverse=True)[:5]
        for name, norm in top_grads:
            print(f"{name}: {norm:.3f}")
        return total_norm

    # Initialize parameters for tracking of the run
    target_sample = []
    prediction_sample = []
    
    # Add memory management parameters
    cleanup_freq = 200  # Regular cleanup every 200 steps
    major_cleanup_freq = 1000  # Major cleanup every 1000 steps
    memory_threshold = 0.8  # Only cleanup if memory usage > 80%
    
    

    for step, batch in enumerate(metric_logger.log_every(data_loader, print_freq, header)):
        if len(batch) == 3:  # If batch contains global view, local view, and mask
            videos, local_videos, bool_masked_pos = batch
            videos = videos.to(device, non_blocking=True)
            local_videos = local_videos.to(device, non_blocking=True)
        else:  # Original case: just global view and mask
            videos, bool_masked_pos = batch
            videos = videos.to(device, non_blocking=True)
            local_videos = None
    
        tokenizer_type = 'default'
        if hasattr(model.module.encoder, 'patch_embed') and hasattr(model.module.encoder.patch_embed, 'tokenizer_type'):
            tokenizer_type = model.module.encoder.patch_embed.tokenizer_type
          
        
        # Mask handling
        if isinstance(bool_masked_pos, (tuple, list)):
            global_bool_mask = bool_masked_pos[0]  # For target/indexing purposes
            full_mask_data = bool_masked_pos      # For the model that needs both global and local masks
            if mask_type == 'multi_local':
                full_mask_data = (bool_masked_pos[0], bool_masked_pos[1])
            else:
                full_mask_data = bool_masked_pos
        else:
            global_bool_mask = bool_masked_pos
            full_mask_data = bool_masked_pos
        
        global_bool_mask = global_bool_mask.to(device, non_blocking=True)
        
        
       
        if len(global_bool_mask.shape) > 2:
            global_bool_mask = global_bool_mask.flatten(1)
        global_bool_mask = global_bool_mask.to(torch.bool)
        
        if grayscale_mode:
            videos = videos.unsqueeze(1)
        bs, _, nf, h, w = videos.shape
        
        # Create Turbo Training mask if enabled
        if hasattr(model.module, 'use_turbo_training') and model.module.use_turbo_training:
            # Create a selective mask for reconstruction
            turbo_full_mask = create_turbo_mask(
                global_bool_mask, 
                turbo_recon_ratio=model.module.turbo_recon_ratio
            )
            
            # Prepare masks for the model
            if isinstance(full_mask_data, (tuple, list)):
                # For multi_local mask type
                turbo_full_mask = (turbo_full_mask, full_mask_data[1])
            else:
                # For tube mask type
                turbo_full_mask = turbo_full_mask
        else:
            # Use original mask for standard training
            turbo_full_mask = None
        
        # Define target generation function
        def generate_targets_for_tokenizer(videos, tokenizer_type='default'):
            """Generate appropriate targets based on tokenizer type"""
            if tokenizer_type == 'sparse_tube':
                # For sparse tubes, directly get tokens from the model's tokenizer
                with torch.no_grad():
                
                    # Get tokens directly from the model's tokenizer
                    tokens = model.module.encoder.patch_embed(videos)
                    
                    # Apply positional embedding if needed (optional)
                    if hasattr(model.module.encoder, 'pos_embed'):
                        pos_embed = model.module.encoder.pos_embed
                        if pos_embed is not None:
                            # Skip cls token position if present
                            if pos_embed.shape[1] > tokens.shape[1]:
                                tokens = tokens + pos_embed[:, 1:, :]
                    
                    if normlize_target:
                        tokens = (tokens - tokens.mean(dim=1, keepdim=True)) / (
                            tokens.var(dim=1, unbiased=True, keepdim=True).sqrt() + 1e-6)
                
                return tokens
            else:
                # Original target generation for standard tokenization
                mean = torch.as_tensor(IMAGENET_DEFAULT_MEAN).to(device)[None, :, None, None, None]
                std = torch.as_tensor(IMAGENET_DEFAULT_STD).to(device)[None, :, None, None, None]
                unnorm_videos = videos * std + mean  # in [0, 1]
                
                if normlize_target:
                    videos_squeeze = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2) c', 
                                              p0=2, p1=patch_size, p2=patch_size)
                    videos_norm = (videos_squeeze - videos_squeeze.mean(dim=-2, keepdim=True)) / (
                        videos_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    videos_patch = rearrange(videos_norm, 'b n p c -> b n (p c)')
                else:
                    videos_patch = rearrange(unnorm_videos, 'b c (t p0) (h p1) (w p2) -> b (t h w) (p0 p1 p2 c)', 
                                            p0=2, p1=patch_size, p2=patch_size)
                
                # if step == 0:
                #     print(f"DEBUG: Standard tokenizer output shape: {videos_patch.shape}")
                
                return videos_patch
        
        # Pixel or MLP target type
        if ('pixel' in target_type) or ('mlp' in target_type):
            with torch.no_grad():
                # Generate targets using the appropriate method
                videos_patch = generate_targets_for_tokenizer(videos, tokenizer_type)
                
                B, N, C = videos_patch.shape
                
                # VALIDATION CHECK: Ensure mask and target dimensions match
                if global_bool_mask.shape[1] != N:
                    print(f"WARNING: Mask shape {global_bool_mask.shape} doesn't match target shape {videos_patch.shape}")
                    print("EMERGENCY FIX: Regenerating mask to match target shape")
                    
                    # Emergency fix: resize mask if needed
                    if tokenizer_type == 'sparse_tube':
                        # For sparse tubes - mask should match token count
                        # Create a new mask with the right dimensions based on the tokenizer output
                        mask_ratio = global_bool_mask.sum() / global_bool_mask.numel()  # Preserve mask ratio
                        new_mask = torch.zeros(B, N, device=device)
                        
                        # Generate random indices to mask
                        for i in range(B):
                            indices = torch.randperm(N, device=device)[:int(mask_ratio * N)]
                            new_mask[i, indices] = 1
                        
                        global_bool_mask = new_mask.to(torch.bool)
                        print(f"DEBUG: Emergency mask fix - new shape: {global_bool_mask.shape}")
                
                try:
                    
                    labels = videos_patch[global_bool_mask].reshape(B, -1, C)
                    # if step == 0:
                    #     print(f"DEBUG: Successfully indexed. Labels shape: {labels.shape}")
                except Exception as e:
                    print(f"ERROR during indexing: {e}")
                    # Alternative approach if indexing fails
                    if tokenizer_type == 'sparse_tube':
                        # print("FALLBACK: Using alternative masking approach for sparse tubes")
                        # Create a new mask with batch dimension
                        batch_indices = torch.arange(B, device=device).view(-1, 1).repeat(1, N//2)
                        token_indices = torch.randperm(N, device=device)[:N//2].view(1, -1).repeat(B, 1)
                        # Extract selected tokens
                        labels = videos_patch[batch_indices, token_indices]
                        # print(f"DEBUG: Fallback labels shape: {labels.shape}")
                    else:
                        raise
                
        # DINO target type
        elif 'dino' in target_type:
            with torch.no_grad():
                if tokenizer_type == 'sparse_tube':
                    # For sparse tubes with DINO targets
                    features = generate_targets_for_tokenizer(videos, tokenizer_type)
                    
                    # Apply feature transformation if needed for DINO
                    if hasattr(model.module, 'feature_projection') and model.module.feature_projection is not None:
                        features = model.module.feature_projection(features)
                else:
                    # Original DINO feature extraction
                    permuted_video = videos.permute(0, 2, 1, 3, 4)
                    bs, nf, _, h, w = permuted_video.shape
                
                    permuted_video = permuted_video[:, ::2].flatten(0, 1)
                    permuted_video = permuted_video.to(device, non_blocking=True)
                    features = feature_extraction_model(permuted_video)
                    _, np, dim = features.shape
                    features = features.reshape(bs, nf//2, np, dim)
                    features.requires_grad = False
                    
                    features = features.to(device, non_blocking=True)
                    features_squeeze = rearrange(features, 'b n o c -> b (n o) c')
                    
                    if normlize_target:
                        features = (features_squeeze - features_squeeze.mean(dim=-2, keepdim=True)) / (
                            features_squeeze.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
                    else:
                        features = features_squeeze
                
                B, N, C = features.shape
                
             
                # VALIDATION CHECK: Ensure mask and target dimensions match
                if global_bool_mask.shape[1] != N:
                    print(f"WARNING: Mask shape {global_bool_mask.shape} doesn't match target shape {features.shape}")
                    print("EMERGENCY FIX: Regenerating mask to match target shape")
                    
                    # Emergency fix: resize mask if needed
                    if tokenizer_type == 'sparse_tube':
                        # For sparse tubes - mask should match token count
                        # Create a new mask with the right dimensions based on the tokenizer output
                        mask_ratio = global_bool_mask.sum() / global_bool_mask.numel()  # Preserve mask ratio
                        new_mask = torch.zeros(B, N, device=device)
                        
                        # Generate random indices to mask
                        for i in range(B):
                            indices = torch.randperm(N, device=device)[:int(mask_ratio * N)]
                            new_mask[i, indices] = 1
                        
                        global_bool_mask = new_mask.to(torch.bool)
                        
                
                    try:
                        # This will now work because the dimensions match
                        labels = features[global_bool_mask].reshape(B, -1, C)
                        # if step == 0:
                        #     print(f"DEBUG: Successfully indexed. Labels shape: {labels.shape}")
                    except Exception as e:
                        print(f"ERROR during indexing: {e}")
                        # Alternative approach if indexing fails
                        if tokenizer_type == 'sparse_tube':
                            # print("FALLBACK: Using alternative masking approach for sparse tubes")
                            # Create a new mask with batch dimension
                            batch_indices = torch.arange(B, device=device).view(-1, 1).repeat(1, N//2)
                            token_indices = torch.randperm(N, device=device)[:N//2].view(1, -1).repeat(B, 1)
                            # Extract selected tokens
                            labels = features[batch_indices, token_indices]
                            # print(f"DEBUG: Fallback labels shape: {labels.shape}")
                        else:
                            raise

        # Continue with the rest of the function as before
        # Prepare forward pass data
        batch_size = videos.shape[0]
        # import pdb; pdb.set_trace()
        if isinstance(full_mask_data, (tuple, list)):
            if mask_type == 'multi_local':
                # Transfer local masks to device (global_bool_mask already on device)
                local_masks = [m.to(device, non_blocking=True) for m in full_mask_data[1]]
                full_mask_data = (global_bool_mask, local_masks)  # global_bool_mask already on device
            else:
                # For non-multi_local tuple/list masks, transfer each component
                full_mask_data = full_mask_data[0].to(device, non_blocking=True)
                # full_mask_data = [m.to(device, non_blocking=True) for m in full_mask_data]
        else:
            # Single mask case
            full_mask_data = full_mask_data.to(device, non_blocking=True)

        with torch.cuda.amp.autocast():
            # Forward pass with Turbo Training if enabled
            if hasattr(model.module, 'use_turbo_training') and model.module.use_turbo_training:
                # Create turbo mask for selective reconstruction
                turbo_full_mask = create_turbo_mask(
                    full_mask_data, 
                    turbo_recon_ratio=getattr(model.module, 'turbo_recon_ratio', 0.25)
                )
                
                # Debug info
                # print(f"\nDEBUG: Step {step} - Using Turbo Training")
                
                # Pass both masks to the model
                outputs, (scores1, q1), (scores2, q2, dino_loss) = model(videos, full_mask_data, labels, epoch, turbo_mask=turbo_full_mask)
                
                # Debug info
                    # print(f"DEBUG: Model outputs shape: {outputs.shape}")
                    # if scores1 is not None:
                    #     print(f"DEBUG: SWAV scores1 shape: {scores1.shape}")
            elif dino_only:
                dino_loss = model(videos, full_mask_data, labels, epoch)
            elif model.module.use_combined_dino_swav:
                if use_video_dino:
                    outputs, (scores1, q1), (scores2, q2, video_dino_loss, image_dino_loss) = model(videos, full_mask_data, labels, epoch,local_videos=local_videos )
                else:
                    outputs, (scores1, q1), (scores2, q2, dino_loss) = model(videos, full_mask_data, labels, epoch)
            else:
                # Standard forward pass
                outputs, (scores1, q1), (scores2, q2, dino_loss) = model(videos, full_mask_data, labels, epoch)

            if loss_func == 'SWAV':
                if dino_only:
                    # Only use DINO loss when dino_only is True
                    # print("DINO loss only mein aaya")
                    loss = dino_weight * dino_loss
                else:
                    # Existing SWAV + DINO loss computation
                    q1 = q1.argmax(dim=-1)
                    scores1 = scores1 / 0.1

                    q2 = q2.argmax(dim=-1)
                    scores2 = scores2 / 0.1

                    swav_loss1 = loss_cpt(scores1.permute(0, 2, 1), q2.long())
                    swav_loss2 = loss_cpt(scores2.permute(0, 2, 1), q1.long())
                    swav_loss = swav_loss1 + swav_loss2
                    
                    # Apply loss weights
                    weighted_swav_loss = swav_weight * swav_loss
                    if not model.module.skip_dino_loss:
                        # print('dino loss',dino_loss)
                        if use_video_dino:
                            # Use image_dino_loss directly since its gradients are already computed
                            # Only weight video DINO loss
                            weighted_dino_loss = video_dino_loss * video_dino_weight + image_dino_loss * dino_weight
                            loss = weighted_swav_loss + weighted_dino_loss 
                        else:   
                            weighted_dino_loss = dino_weight * dino_loss 
                            loss = weighted_swav_loss + weighted_dino_loss
                            # print('weighted dino loss',weighted_dino_loss)
                        # print('weighted dino loss',weighted_dino_loss)
                    else:
                        loss = weighted_swav_loss
                    
                    # Add loss debugging prints
                    # print("\n=== Loss Computation ===")
                    # print(f"Weighted SWAV Loss: {weighted_swav_loss.item():.4f}")
                    # print(f"Weighted DINO Loss: {weighted_dino_loss.item():.4f}")
                    # print(f"Total Loss: {loss.item():.4f}")
            else:
                # For Turbo Training with non-SWAV loss
                if hasattr(model.module, 'use_turbo_training') and model.module.use_turbo_training:
                    # Extract targets using the turbo mask
                    tokenizer_type = getattr(model.module.encoder, 'tokenizer_type', 'default')
                    videos_patch = generate_targets_for_tokenizer(videos, tokenizer_type)
                    
                    # Debug info - handle tuple/list case properly
                    # print(f"DEBUG: videos_patch shape: {videos_patch.shape}")
                    
                    if isinstance(turbo_full_mask, (tuple, list)):
                       
                        turbo_mask_for_indexing = turbo_full_mask[0]  # Use the first element for indexing
                    else:
                        # print(f"DEBUG: turbo_full_mask shape: {turbo_full_mask.shape}, sum: {turbo_full_mask.sum().item()}")
                        turbo_mask_for_indexing = turbo_full_mask
                    
                    # Safety check to avoid empty selection
                    if turbo_mask_for_indexing.sum().item() == 0:
                        # print("WARNING: Empty turbo mask! Using random tokens instead.")
                        # Create a random mask with ~1% of tokens
                        B, N = videos_patch.shape[:2]
                        random_mask = torch.zeros(B, N, device=videos_patch.device)
                        for b in range(B):
                            indices = torch.randperm(N, device=videos_patch.device)[:max(1, int(0.01 * N))]
                            random_mask[b, indices] = 1
                        turbo_mask_for_indexing = random_mask.to(torch.bool)
                        # print(f"DEBUG: Random mask created with {turbo_mask_for_indexing.sum().item()} tokens")
                    
                    try:
                        # Extract targets using the turbo mask
                        B, N, C = videos_patch.shape
                        turbo_labels = videos_patch[turbo_mask_for_indexing].reshape(B, -1, C)
                        # print(f"DEBUG: turbo_labels shape: {turbo_labels.shape}")
                        
                        # Compute reconstruction loss on the subset of tokens
                        turbo_loss = loss_cpt(outputs, turbo_labels).mean()
                        # print(f"DEBUG: Reconstruction loss: {turbo_loss.item():.4f}")
                        
                        # Use the turbo_weight parameter from function arguments
                        weighted_turbo_loss = turbo_loss * turbo_weight
                        # print(f"DEBUG: Weighted turbo loss ({turbo_weight} * {turbo_loss.item():.4f} = {weighted_turbo_loss.item():.4f})")
                        
                        # Start with weighted turbo loss
                        loss = weighted_turbo_loss
                        
                        # Add DINO loss if not skipped
                        if not model.module.skip_dino_loss:
                            weighted_dino_loss = dino_weight * dino_loss
                            loss = loss + weighted_dino_loss
                            # print(f"DEBUG: With DINO loss: {loss.item():.4f}")
                    except Exception as e:
                          
                            # Fallback to standard training
                          
                            loss = loss_cpt(outputs, labels).mean()
                            turbo_loss = loss  # Set turbo_loss for logging
                else:
                            # Standard loss computation (unchanged)
                        loss = loss_cpt(outputs, labels).mean()
                        turbo_loss = loss  # Set turbo_loss for logging
                
        loss_value = loss.item()

        if not math.isfinite(loss_value):
            print("Loss is {}, stopping training".format(loss_value))
            sys.exit(1)

        optimizer.zero_grad()
        
        # Add gradient clipping and NaN checks
        def check_gradients(model):
            has_grads = False
            for name, param in model.named_parameters():
                if param.grad is not None:
                    has_grads = True
                    if torch.isnan(param.grad).any():
                        print(f"NaN gradient detected in {name}")
                        return False
            return has_grads  # Only return True if we actually checked some gradients

        # this attribute is added by timm on one optimizer (adahessian)
        is_second_order = hasattr(optimizer, 'is_second_order') and optimizer.is_second_order
        grad_norm = loss_scaler(loss, optimizer, clip_grad=max_norm,
                                parameters=model.parameters(), create_graph=is_second_order)
        
        # Check for NaN gradients after backward pass
        if not check_gradients(model):
            print("NaN gradients detected or no gradients computed, skipping step")
            optimizer.zero_grad()
            continue
            
        # Clip gradients to prevent explosion
        if max_norm is not None:
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm)
        
        loss_scale_value = loss_scaler.state_dict()["scale"]

        # if step % print_freq == 0:
        #     print("\n=== Post-backward Grad Check ===")
        #     print_grad_norms(model, "after backward")
        #     print(f"Clipped grad norm: {grad_norm:.3f}")

        torch.cuda.synchronize()
        
        # Regular memory cleanup - minimal impact
        if step % cleanup_freq == 0:
            # Check memory usage first
            if torch.cuda.memory_allocated() / torch.cuda.max_memory_allocated() > memory_threshold:
                # Clear only critical tensors
                if 'features' in locals():
                    del features
                if 'labels' in locals():
                    del labels
                if 'videos_patch' in locals():
                    del videos_patch
                torch.cuda.empty_cache()

        # Major cleanup - less frequent
        if step % major_cleanup_freq == 0:
            # Clear CUDA cache
            torch.cuda.empty_cache()
            
            # Clear intermediate variables
            for name in ['features', 'labels', 'videos_patch', 'outputs', 
                       'permuted_video', 'features_squeeze']:
                if name in locals():
                    del locals()[name]
            
            # Run garbage collection
            import gc
            gc.collect()

        # Before the metric updates, initialize all possible metrics
        metric_logger.update(loss=loss_value)
        metric_logger.update(loss_scale=loss_scale_value)
        min_lr = 10.
        max_lr = 0.
        for group in optimizer.param_groups:
            min_lr = min(min_lr, group["lr"])
            max_lr = max(max_lr, group["lr"])

        metric_logger.update(lr=max_lr)
        metric_logger.update(min_lr=min_lr)
        weight_decay_value = None
        for group in optimizer.param_groups:
            if group["weight_decay"] > 0:
                weight_decay_value = group["weight_decay"]
        metric_logger.update(weight_decay=weight_decay_value if weight_decay_value is not None else 0.0)
        metric_logger.update(grad_norm=grad_norm)

        # Update loss-specific metrics with safe defaults
        if (loss_func == 'SWAV' and not dino_only):
            # Case 1: Regular SWAV + DINO training
            if not model.module.skip_dino_loss and not use_video_dino:
                metric_logger.update(dino_loss=dino_loss.item())
            metric_logger.update(swav_loss=swav_loss.item())
        if model.module.use_combined_dino_swav:
            # Case 3: Video DINO training
            if use_video_dino:
                metric_logger.update(video_dino_loss=video_dino_loss.item())
                metric_logger.update(image_dino_loss=image_dino_loss.item())
            else:
                metric_logger.update(image_dino_loss=dino_loss.item())
        elif loss_func == 'L2':
            # Case 4: L2 training
            if model.module.use_turbo_training:
                # Case 4a: L2 with Turbo training
                metric_logger.update(turbo_loss=dino_loss.item())
            else:
                # Case 4b: Regular L2 training
                metric_logger.update(dino_loss=dino_loss.item())
        elif dino_only:
            # Case 5: DINO only training
            metric_logger.update(dino_loss=dino_loss.item())
        if log_writer is not None:
            if dino_only or loss_func == 'L2':
                log_dict = {
                    "mae_Loss": loss_value,
                    "DINO_Loss": dino_loss.item(),
                    "Lr_max": max_lr,
                    "Lr_min": min_lr,
                "Weight_decay": weight_decay_value,
                "Grad_norm": grad_norm,}
            elif model.module.use_combined_dino_swav:
                
                log_dict = {
                    "mae_Loss": loss_value,
                    "SWAV_Loss": swav_loss.item(),
                    "Lr_max": max_lr,
                    "Lr_min": min_lr,
                    "Weight_decay": weight_decay_value,
                    "Grad_norm": grad_norm,
                }
                if use_video_dino:
                    log_dict.update({
                        "Video_DINO_Loss": video_dino_loss.item(),
                    })
                else:
                    log_dict.update({
                        "Image_DINO_Loss": dino_loss.item(),
                    })
            else:
                log_dict = {
                    "mae_Loss": loss_value,
                    "DINO_Loss": dino_loss.item(),
                    "Lr_max": max_lr,
                    "Lr_min": min_lr,
                    "Weight_decay": weight_decay_value,
                    "Grad_norm": grad_norm,
                }
            # Only add turbo-related metrics if turbo training is enabled
            if hasattr(model.module, 'use_turbo_training') and model.module.use_turbo_training:
                log_dict.update({
                    "Turbo_Loss": dino_loss.item(),
                    "Weighted_Turbo_Loss": dino_loss.item(),
                    "Turbo_Weight": turbo_weight
                })
            
            log_writer.log(log_dict)

        if lr_scheduler is not None:
            lr_scheduler.step_update(start_steps + step)
        
        # Check tokenizer type before using patch_size
        tokenizer_type = getattr(model.module.encoder.patch_embed, 'tokenizer_type', 'default')
        
        # Debug information about tokenizer
        # print(f"DEBUG: Tokenizer type: {tokenizer_type}")
        
        # Only access kernel_sizes if it's a sparse tube tokenizer
        # if tokenizer_type == 'sparse_tube':
        #     print(f"  - Kernel sizes: {model.module.encoder.patch_embed.kernel_sizes}")
        #     print(f"  - Strides: {model.module.encoder.patch_embed.strides}")
        #     print(f"  - Offsets: {model.module.encoder.patch_embed.offsets}")
        # else:
        #     print(f"  - Patch size: {patch_size}")
        
        # if step == 0 and epoch == 0:
        #     # Print model details on first step of first epoch for debugging
        #     encoder_info = model.module.encoder
        #     if hasattr(encoder_info, 'patch_embed') and hasattr(encoder_info.patch_embed, 'strides'):
        #         print(f"Using sparse tube tokenizer with:")
        #         print(f"  - Kernel sizes: {encoder_info.patch_embed.kernel_sizes}")
        #         print(f"  - Strides: {encoder_info.patch_embed.strides}")
        #         print(f"  - Offsets: {encoder_info.patch_embed.offsets}")
        #         if isinstance(bool_masked_pos, tuple) and len(bool_masked_pos) > 1:
        #             print(f"Using multi-scale masking with {len(bool_masked_pos)} mask components")
        #         print(f"Position embedding shape: {encoder_info.pos_embed.shape}")

        # # Add debug output after model runs
        # if step == 0:  # Print only on first step
        #     # print(f"DEBUG: Model outputs keys: {outputs.keys()}")
        #     if 'pred' in outputs:
        #         print(f"DEBUG: Prediction shape: {outputs['pred'].shape}")
        #     if 'mask' in outputs:
        #         print(f"DEBUG: Mask shape in output: {outputs['mask'].shape}")
        
        #     # Check shape of labels and mask before the problematic line
        #     try:
        #         if 'labels' in locals():
        #             print(f"DEBUG: Labels shape: {labels.shape}")
        #         if 'global_bool_mask' in locals():
        #             print(f"DEBUG: global_bool_mask shape: {global_bool_mask.shape}")
        #     except:
        #         print("DEBUG: Labels or global_bool_mask not defined yet")

        # For first few batches, print shapes and device info
        # if step < 3:
        #     print(f"\nStep {step}")
        #     print(f"Input batch shape: {batch.shape}")
        #     print(f"Input device: {batch.device}")

        # After forward pass
        # print(f"\nStep {step} losses:")
        # print(f"DINO loss: {dino_loss.item():.4f}")
        # print(f"Total loss: {loss.item():.4f}")

        # After optimizer step
        # if step < 3:
        #     print(f"Grad norms after backward:")
        #     for name, param in model.named_parameters():
        #         if param.grad is not None:
        #             print(f"{name}: {param.grad.norm().item():.4f}")

        # if model.module.use_combined_dino_swav:
        #     print(f"\nStep {step} - Combined mode active")
        #     print(f"Losses - SWAV: {swav_loss.item():.4f}, DINO: {dino_loss.item():.4f}")

        # Add after processing each batch
        torch.cuda.empty_cache()
        if 'features' in locals(): del features
        if 'similarities' in locals(): del similarities

    # End of epoch cleanup
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    
    # gather the stats from all processes
    metric_logger.synchronize_between_processes()
    print("Averaged stats:", metric_logger)
    return {k: meter.global_avg for k, meter in metric_logger.meters.items()}

def create_turbo_mask(original_mask, turbo_recon_ratio=0.25):
    """
    Create a selective mask for Turbo Training reconstruction.
    Vectorized implementation for better performance.
    
    Args:
        original_mask: The original mask used for encoding
        turbo_recon_ratio: Ratio of masked tokens to reconstruct
        
    Returns:
        turbo_mask: Selective mask for reconstruction
    """
    # Handle tuple/list case
    if isinstance(original_mask, (tuple, list)):
        global_mask = original_mask[0]
        has_local_mask = True
        
        # Check if second element is an empty list
        second_element = original_mask[1] if len(original_mask) > 1 else None
        is_empty_list = isinstance(second_element, list) and len(second_element) == 0
        # print(f"DEBUG: Second element is empty list: {is_empty_list}")
    else:
        global_mask = original_mask
        has_local_mask = False
        is_empty_list = False
    
    # Ensure mask is boolean
    global_mask = global_mask.to(torch.bool)
    
    # Get dimensions
    B = global_mask.shape[0]
    device = global_mask.device
    
    # Create empty turbo mask with same shape as global mask
    turbo_mask = torch.zeros_like(global_mask)
    
    # Track total tokens for fallback mechanism
    total_tokens = global_mask.numel()
    total_masked = global_mask.sum().item()
    
    # Vectorized implementation
    # For each sample in the batch:
    for b in range(B):
        # Get indices of masked tokens
        masked_indices = torch.nonzero(global_mask[b]).squeeze(1)
        
        # Skip if no tokens are masked
        if masked_indices.numel() == 0:
            continue
            
        # Calculate number of tokens to reconstruct
        num_masked = masked_indices.size(0)
        num_to_recon = max(1, min(int(turbo_recon_ratio * num_masked), num_masked))
        
        # Randomly select indices to reconstruct
        perm = torch.randperm(num_masked, device=device)[:num_to_recon]
        selected_indices = masked_indices[perm]
        
        # Set selected indices to 1 in turbo mask
        turbo_mask[b, selected_indices] = 1
    
    # Debug info
    total_selected = turbo_mask.sum().item()
    # print(f"DEBUG: Selected for reconstruction: {total_selected} tokens ({total_selected/total_masked:.2%} of masked)")
    
    # Ensure we have at least some tokens selected
    if total_selected == 0:
        # print("WARNING: No tokens selected for reconstruction! Falling back to random selection.")
        # Create a random mask with ~1% of tokens
        random_indices = torch.randperm(total_tokens, device=device)[:max(1, int(0.01 * total_tokens))]
        flat_mask = turbo_mask.view(-1)
        flat_mask[random_indices] = 1
        # print(f"DEBUG: Fallback selected {turbo_mask.sum().item()} tokens")
    
    # Return in the same format as original_mask
    if has_local_mask:
        if is_empty_list:
            # If second element was an empty list, keep it that way
            return (turbo_mask, [])
        else:
            # Otherwise, keep the original second element
            return (turbo_mask, original_mask[1])
    else:
        return turbo_mask

def knn_evaluation(model: torch.nn.Module, 
                  val_loader: Iterable,
                  test_loader: Iterable,
                  device: torch.device,
                  k: int = 20,
                  temperature: float = 0.07):
    """
    Perform KNN evaluation using validation set for fitting and test set for evaluation.
    Returns multiple metrics with macro averaging.
    """
    model.eval()
    
    # Extract validation features (for fitting)
    val_features = []
    val_labels = []
    
    print("Extracting validation features...")
    with torch.no_grad():
        for batch in val_loader:
            videos, labels = batch[0], batch[1]
            videos = videos.to(device)
            dummy_mask = torch.zeros((videos.shape[0], model.module.encoder.patch_embed.num_patches), 
                                   dtype=torch.bool,
                                   device=videos.device)
            
            features = model.module.encoder(
                videos, 
                dummy_mask,
                is_dino=True
            )[0]
            
            # Get CLS token or mean pooling
            if not model.module.skip_dino_loss:
                features = features[:, 0]  # CLS token
            else:
                features = features.mean(dim=1)  # mean pooling
                
            features = F.normalize(features, dim=1)
            val_features.append(features.cpu())
            val_labels.append(labels.cpu())
    
    val_features = torch.cat(val_features, dim=0)
    val_labels = torch.cat(val_labels, dim=0)
    del features
    del videos
    del labels
    del dummy_mask
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    # For collecting all predictions and labels
    all_predictions = []
    all_labels = []
    
    print("Evaluating on test set...")
    with torch.no_grad():
        for batch in test_loader:
            videos, labels = batch[0], batch[1]
            videos = videos.to(device)
            dummy_mask = torch.zeros((videos.shape[0], model.module.encoder.patch_embed.num_patches), 
                                   dtype=torch.bool,
                                   device=videos.device)
            
            features = model.module.encoder(
                videos, 
                dummy_mask,
                is_dino=True
            )[0]
            
            # Get CLS token or mean pooling
            if not model.module.skip_dino_loss:
                features = features[:, 0]
            else:
                features = features.mean(dim=1)
                
            features = F.normalize(features, dim=1)
            
            # Compute similarities
            similarities = torch.mm(features.cpu(), val_features.t())
            similarities = similarities / temperature
            
            # Find top-k neighbors
            _, indices = similarities.topk(k, dim=1)
            neighbors_labels = val_labels[indices]
            
            # Weighted voting
            predictions = torch.zeros(features.size(0), val_labels.max() + 1)
            predictions.scatter_add_(1, neighbors_labels, 
                                   torch.exp(similarities[torch.arange(features.size(0)).unsqueeze(1), indices]))
            
            # Get predicted labels
            pred_labels = predictions.argmax(dim=1)
            
            # Collect predictions and labels
            all_predictions.extend(pred_labels.cpu().numpy())
            all_labels.extend(labels.cpu().numpy()) 
    
    # Convert to numpy arrays
    all_predictions = np.array(all_predictions)
    all_labels = np.array(all_labels)
    
    del features, similarities
    del val_features, val_labels
    del videos, labels
    del dummy_mask
    torch.cuda.empty_cache()
    import gc
    gc.collect()
    # Calculate metrics
    accuracy = (all_predictions == all_labels).mean()
    balanced_acc = balanced_accuracy_score(all_labels, all_predictions)
    f1 = f1_score(all_labels, all_predictions, average='macro')
    precision = precision_score(all_labels, all_predictions, average='macro')
    recall = recall_score(all_labels, all_predictions, average='macro')
    
    # Print all metrics
    print(f"=== KNN Evaluation Metrics ===")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Balanced Accuracy: {balanced_acc:.4f}")
    print(f"Macro F1-score: {f1:.4f}")
    print(f"Macro Precision: {precision:.4f}")
    print(f"Macro Recall: {recall:.4f}")
    
    # Return all metrics in a dictionary
    metrics = {
        'accuracy': accuracy,
        'balanced_accuracy': balanced_acc,
        'f1_score': f1,
        'precision': precision,
        'recall': recall
    }
    
    return metrics
