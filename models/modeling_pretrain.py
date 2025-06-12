import math
import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.utils.checkpoint as checkpoint
from functools import partial
from discovr.utils import utils
from discovr.models.modeling_finetune import Block, _cfg, PatchEmbed, get_sinusoid_encoding_table
from timm.models.registry import register_model
from timm.models.layers import trunc_normal_ as __call_trunc_normal_
from discovr.models.attn_mask import create_binary_kk_attention_mask_2d
import copy
import torch.distributed as dist
from discovr.models.TubeViT.tubevit.model import SparseTubesTokenizer
from discovr.models.TubeViT.tubevit.positional_encoding import get_3d_sincos_pos_embed
import numpy as np
from einops import rearrange


def trunc_normal_(tensor, mean=0., std=1.):
    __call_trunc_normal_(tensor, mean=mean, std=std, a=-std, b=std)


__all__ = [
    'pretrain_videomae_small_patch16_224',
    'pretrain_videomae_base_patch16_224', 
    'pretrain_videomae_large_patch16_224', 
    'pretrain_videomae_huge_patch16_224',
]


class PretrainVisionTransformerEncoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, img_size=224, patch_size=16, in_chans=3, num_classes=0, embed_dim=768, depth=12,
                 num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0.,
                 drop_path_rate=0., norm_layer=nn.LayerNorm, init_values=None, tubelet_size=2, use_checkpoint=False,
                 mask_type='tube', use_learnable_pos_emb=False, num_frames=16, tokenizer_type='default',use_mean_pooling=False,fc_drop_rate=0.,skip_dino_loss=False, use_dino_crop=False,local_size=96,num_local_views=4):
        super().__init__()
        print(f"PretrainVisionTransformerEncoder init - local_size: {local_size}")
        self.num_classes = num_classes
        self.num_features = self.embed_dim = embed_dim
        self.num_frames = num_frames
        self.tokenizer_type = tokenizer_type
        self.img_size = img_size
        self.skip_dino_loss = skip_dino_loss
        self.use_dino_crop = use_dino_crop
        self.local_size = local_size
        self.num_local_views = num_local_views
        # Initialize appropriate tokenizer
        if tokenizer_type == 'default':
            print('Image_size',self.img_size)
            self.patch_embed = PatchEmbed(
                img_size=self.img_size, 
                patch_size=patch_size, 
                in_chans=in_chans, 
                embed_dim=embed_dim,
                tubelet_size=tubelet_size,
                num_frames=num_frames
            )
            num_patches = self.patch_embed.num_patches
            # print('DEFAULT_MEIN_AAYA',num_patches)
            # Set patch_size attribute here - it's modifiable for default tokenizer
            self.patch_embed.patch_size = patch_size
            
        elif tokenizer_type == 'sparse_tube':  # sparse_tube
            # print('SPARSE_TUBE_MEIN_AAYA')
            # Define kernel sizes, strides, and offsets (from TubeViT implementation)
            kernel_sizes = (
                (8, 8, 8),
                (16, 4, 4),
                (4, 12, 12),
                (1, 16, 16),
            )
            strides = (
                (16, 32, 32),
                (6, 32, 32),
                (16, 32, 32),
                (32, 16, 16),
            )
            offsets = (
                (0, 0, 0),
                (4, 8, 8),
                (0, 16, 16),
                (0, 0, 0),
            )
            
            self.patch_embed = SparseTubesTokenizer(
                hidden_dim=embed_dim,
                kernel_sizes=kernel_sizes,
                strides=strides,
                offsets=offsets
            )
            
            # Save the configuration for positional encoding
            self.patch_embed.video_shape = (3, num_frames, img_size, img_size)
            self.patch_embed.tokenizer_type = 'sparse_tube'  # Add type marker to embedder
            
            # Calculate token counts correctly
            num_frames = self.num_frames
            height = img_size
            width = img_size
            
            # Calculate token counts for each tube type
            tube_token_counts = []
            total_tokens = 0
            
            for kernel, stride, offset in zip(self.patch_embed.kernel_sizes, 
                                             self.patch_embed.strides, 
                                             self.patch_embed.offsets):
                # Calculate tokens using proper formula
                t_tokens = max(1, (num_frames - offset[0] - kernel[0] + 1 + stride[0] - 1) // stride[0])
                h_tokens = max(1, (height - offset[1] - kernel[1] + 1 + stride[1] - 1) // stride[1])
                w_tokens = max(1, (width - offset[2] - kernel[2] + 1 + stride[2] - 1) // stride[2])
                
                token_count = t_tokens * h_tokens * w_tokens
                tube_token_counts.append(token_count)
                total_tokens += token_count
            
            # Now use total_tokens as the actual number of patches
            num_patches = total_tokens
            self.patch_embed.num_patches = num_patches
            print(f"INFO: Calculated {num_patches} tokens for sparse tube positional encoding")

        self.use_checkpoint = use_checkpoint
        self.mask_type = mask_type

        # Add CLS token
        if not self.skip_dino_loss:
            self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
            trunc_normal_(self.cls_token, std=.02)
        
        # Initialize position embeddings differently based on tokenizer type
        if use_learnable_pos_emb:
            if not self.skip_dino_loss:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim), requires_grad=True)  # +1 for cls token
            else:
                self.pos_embed = nn.Parameter(torch.zeros(1, num_patches, embed_dim), requires_grad=True)  # +1 for cls token
            trunc_normal_(self.pos_embed, std=.02)
        else:
            # Fixed position embedding
            if tokenizer_type == 'sparse_tube':
                # For sparse tubes, we need to create position embeddings for each tube type separately
                pos_embeds = []
                
                # Create a CLS token embedding (zeros)
                if not self.skip_dino_loss:
                    cls_pos_embed = np.zeros([1, embed_dim], dtype=np.float32)
                    pos_embeds.append(cls_pos_embed)
                    print(f"DEBUG: Added CLS token embedding: shape={cls_pos_embed.shape}")
                
                tube_token_count = 0  # Track total tokens for verification
                
                # For each tube configuration, generate position embeddings
                for i, (kernel, stride, offset) in enumerate(zip(self.patch_embed.kernel_sizes, 
                                                               self.patch_embed.strides, 
                                                               self.patch_embed.offsets)):
                    # Calculate tube dimensions
                    t_size = max(1, (num_frames - offset[0] - kernel[0] + 1 + stride[0] - 1) // stride[0])
                    h_size = max(1, (img_size - offset[1] - kernel[1] + 1 + stride[1] - 1) // stride[1])
                    w_size = max(1, (img_size - offset[2] - kernel[2] + 1 + stride[2] - 1) // stride[2])
                    
                    # Create shape tuple for this tube type
                    tube_shape = (t_size, h_size, w_size)
                    tube_tokens = t_size * h_size * w_size
                    tube_token_count += tube_tokens
                    
                    print(f"DEBUG: Tube {i+1}: dimensions={tube_shape}, tokens={tube_tokens}")
                    print(f"DEBUG: Parameters: stride={stride}, offset={offset}, kernel={kernel}")
                    
                    # Generate position embeddings for this tube configuration
                    tube_pos_embed = get_3d_sincos_pos_embed(
                        embed_dim, 
                        tube_shape,
                        stride=stride, 
                        offset=offset, 
                        kernel_size=kernel,
                        cls_token=False  # No cls token for individual tubes
                    )
                    
                    # print(f"DEBUG: Tube {i+1} pos embed shape: {tube_pos_embed.shape}")
                    pos_embeds.append(tube_pos_embed)
                    
                # Concatenate all position embeddings
                pos_embed = np.concatenate(pos_embeds, axis=0)
                
                # print(f"DEBUG: Final combined pos embed shape: {pos_embed.shape}")
                # print(f"DEBUG: Total tokens calculated: {tube_token_count} (+ 1 CLS token)")
                # print(f"DEBUG: Expected total tokens: {num_patches} (+ 1 CLS token)")
                
                if pos_embed.shape[0] != num_patches + 1:
                    print(f"WARNING: Mismatch between pos embed size {pos_embed.shape[0]} and expected token count {num_patches + 1}")
                
            else:
                # Default tokenizer - use standard positional encoding
                pos_embed = get_sinusoid_encoding_table(num_patches, embed_dim)
                # print('POS_EMBED',pos_embed.shape)
                if not self.skip_dino_loss:
                    cls_pos_embed = torch.zeros(1, 1, embed_dim)
                    pos_embed = torch.cat([cls_pos_embed, pos_embed], dim=1)
                # print('POS_EMBED_WITH_CLS',pos_embed.shape)
            # Directly use the tensor
            if self.tokenizer_type == 'sparse_tube':
                self.pos_embed = nn.Parameter(torch.from_numpy(pos_embed).float().unsqueeze(0), requires_grad=False)
                # print('SPARSE_TUBE_POS_EMBED',self.pos_embed.shape)
            else:
                self.pos_embed = nn.Parameter(pos_embed, requires_grad=False)
               

                # print('DEFAULT_POS_EMBED',self.pos_embed.shape)

        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm = nn.Identity() if use_mean_pooling else norm_layer(embed_dim)
        self.fc_norm = norm_layer(embed_dim) if use_mean_pooling else None
        self.fc_dropout = nn.Dropout(p=fc_drop_rate) if fc_drop_rate > 0 else nn.Identity()
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)

        # Add normalization flag to match target processing
        self.normalize_features = True  # Set this based on your args.normalize_target

        # After the existing pos_embed initialization
        # print('USE_DINO_CROP',self.use_dino_crop)
        # print('USE_LEARNABLE_POS_EMB',use_learnable_pos_emb)
        if self.use_dino_crop and not use_learnable_pos_emb:
            # print(f"Creating local_pos_embed with local_size: {self.local_size}")
            # Calculate number of patches for local views
            local_num_patches = (self.local_size // patch_size) * (self.local_size // patch_size)
            local_video_num_patches = (self.local_size // patch_size) * (self.local_size // patch_size) * num_frames//2
            # print(f"local_num_patches: {local_num_patches}, local_video_num_patches: {local_video_num_patches}")
            
            # Create position embeddings for local views
            local_pos_embed = get_sinusoid_encoding_table(local_num_patches, embed_dim)
            local_pos_embed_video = get_sinusoid_encoding_table(local_video_num_patches, embed_dim)
            
            # Add CLS token if needed
            if not skip_dino_loss:
                cls_pos_embed = torch.zeros(1, 1, embed_dim)
                local_pos_embed = torch.cat([cls_pos_embed, local_pos_embed], dim=1)
                local_pos_embed_video = torch.cat([cls_pos_embed, local_pos_embed_video], dim=1)
            
            # Initialize as non-trainable parameters
            self.local_pos_embed = nn.Parameter(local_pos_embed, requires_grad=False)
            self.local_pos_embed_video = nn.Parameter(local_pos_embed_video, requires_grad=False)
            
            # Initialize with trunc normal
            trunc_normal_(self.local_pos_embed, std=.02)
            trunc_normal_(self.local_pos_embed_video, std=.02)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head
    

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward_features(self, x, mask, is_dino=False):
        # Handle different input formats for image vs video
        
        # For video path: [B, C, T, H, W]
        # print('Input Forward Features',x.shape)
        B, _, T, _, _ = x.shape
        # print(f"Video input shape: {x.shape}")
        mask_pos = None
        # Patch embedding
        x = self.patch_embed(x)
        # print('PATCH_EMBED',x.shape)
        # print(f"Post-patch embed shape: {x.shape}")
        if not self.skip_dino_loss:
            # Add CLS token
            cls_token = self.cls_token.expand(B, -1, -1)
            x = torch.cat((cls_token, x), dim=1)
        
        # Position embedding
        # print('POS_EMBED',self.pos_embed.shape)
        # print('X',x.shape)
        x = x + self.pos_embed.type_as(x).clone().detach()
        
        B, _, C = x.shape
        
        if self.mask_type == 'tube':
            if not self.skip_dino_loss:
                cls_mask = torch.zeros((B, 1), dtype=mask.dtype, device=mask.device)
                # print('CLS_MASK',cls_mask.shape)
                # print('MASK',mask.shape)
                full_mask = torch.cat((cls_mask, mask), dim=1)
            else:
                full_mask = mask
            x_vis = x[~full_mask].reshape(B, -1, C)  # ~mask means visible
        else:
            if not self.skip_dino_loss:
                cls_mask = torch.zeros((B, 1), dtype=mask.dtype, device=mask.device)
                full_mask = torch.cat((cls_mask, mask), dim=1)
            else:
                full_mask = mask
            x_vis = x[~full_mask].reshape(B, -1, C)  # ~mask means visible
        
        if self.use_checkpoint:
            for blk in self.blocks:
                x_vis = checkpoint.checkpoint(blk, x_vis)
        else:   
            for blk in self.blocks:
                x_vis = blk(x_vis)

        x_vis = self.norm(x_vis)
        return x_vis, mask_pos

    def forward(self, x, mask,is_dino=False):
        x, mask_pos = self.forward_features(x, mask,is_dino)
        x = self.head(x)
        return x, mask_pos

class PretrainVisionTransformerDecoder(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self, patch_size=16, num_classes=768, embed_dim=768, depth=12, num_heads=12, mlp_ratio=4.,
                 qkv_bias=False, qk_scale=None, drop_rate=0., attn_drop_rate=0., drop_path_rate=0.,
                 norm_layer=nn.LayerNorm, init_values=None, num_patches=196, tubelet_size=2, use_checkpoint=False,
                 num_frames=16):
        super().__init__()
        self.num_classes = num_classes
        #assert num_classes == 3 * tubelet_size * patch_size ** 2 
        self.num_features = self.embed_dim = embed_dim  # num_features for consistency with other models
        self.patch_size = patch_size
        self.use_checkpoint = use_checkpoint
        self.num_frames = num_frames  # Store num_frames


        dpr = [x.item() for x in torch.linspace(0, drop_path_rate, depth)]  # stochastic depth decay rule
        self.blocks = nn.ModuleList([
            Block(
                dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio, qkv_bias=qkv_bias, qk_scale=qk_scale,
                drop=drop_rate, attn_drop=attn_drop_rate, drop_path=dpr[i], norm_layer=norm_layer,
                init_values=init_values)
            for i in range(depth)])
        self.norm =  norm_layer(embed_dim)
        self.head = nn.Linear(embed_dim, num_classes) if num_classes > 0 else nn.Identity()

        self.apply(self._init_weights)


    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token'}

    def get_classifier(self):
        return self.head

    def reset_classifier(self, num_classes, global_pool=''):
        self.num_classes = num_classes
        self.head = nn.Linear(self.embed_dim, num_classes) if num_classes > 0 else nn.Identity()

    def forward(self, x, return_token_num):
        # print(f"Decoder input shape: {x.shape}")
        
        if self.use_checkpoint:
            for blk in self.blocks:
                x = checkpoint.checkpoint(blk, x)
        else:   
            for blk in self.blocks:
                x = blk(x)
        
        # print(f"After blocks shape: {x.shape}")
        
        if return_token_num > 0:
            x = self.head(self.norm(x[:, -return_token_num:])) 
        else:
            x = self.head(self.norm(x))
        
        # print(f"Decoder output shape: {x.shape}")
        return x


class FeatureExtractor(torch.nn.Module):
    def __init__(self, vit_model, input_size, patch_size):
        super(FeatureExtractor, self).__init__()
        self.vit_model = vit_model
        self.input_size = input_size
        self.patch_size = patch_size
        self.spatial_resolution = input_size // patch_size
        assert self.spatial_resolution * patch_size == input_size

    def forward(self, x):
        if self.patch_size == 14:
            features = self.vit_model.forward_features(x)[:, 5:]
            bs, np, dim = features.shape
            features = features.reshape(bs, self.spatial_resolution, self.spatial_resolution, dim).permute(0, 3, 1, 2)
            features = features.flatten(2, -1).permute(0, 2, 1)
        else:
            features = self.vit_model.forward_features(x)[:, 1:]
        return features

class DINOHead(nn.Module):
    def __init__(self, in_dim, out_dim, hidden_dim=2048, bottleneck_dim=256, use_bn=False, norm_last_layer=True):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.GELU(),
            nn.Linear(hidden_dim, bottleneck_dim),
        )
        self.apply(self._init_weights)
        self.last_layer = nn.utils.weight_norm(nn.Linear(bottleneck_dim, out_dim, bias=False))
        self.last_layer.weight_g.data.fill_(1)
        if norm_last_layer:
            self.last_layer.weight_g.requires_grad = False

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            trunc_normal_(m.weight, std=.02)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.mlp(x)
        x = nn.functional.normalize(x, dim=-1, p=2)
        x = self.last_layer(x)
        return x

class CrossModalAttention(nn.Module):
    def __init__(self, dim, num_heads=8, dropout=0.1):
        super().__init__()
        self.attention = nn.MultiheadAttention(dim, num_heads=num_heads, dropout=dropout)
        self.norm1 = nn.LayerNorm(dim)
        self.norm2 = nn.LayerNorm(dim)
        self.dropout = nn.Dropout(dropout)
        
    def forward(self, image_features, video_features):
        # image_features: [B, N, D]
        # video_features: [B, M, D]
        
        # Store original shapes
        B, N, D = image_features.shape
        M = video_features.shape[1]
        
        # Reshape for attention
        image_features = image_features.permute(1, 0, 2)  # [N, B, D]
        video_features = video_features.permute(1, 0, 2)  # [M, B, D]
        
        # Apply cross-attention
        attended_features, _ = self.attention(
            query=image_features,
            key=video_features,
            value=video_features
        )
        
        # Add residual connection and normalize
        attended_features = self.norm1(image_features + self.dropout(attended_features))
        
        # Reshape back
        attended_features = attended_features.permute(1, 0, 2)  # [B, N, D]
        
        return attended_features

class PretrainVisionTransformer(nn.Module):
    """ Vision Transformer with support for patch or hybrid CNN input stage
    """
    def __init__(self,
                 img_size=224, 
                 patch_size=16, 
                 encoder_in_chans=3, 
                 encoder_num_classes=0, 
                 encoder_embed_dim=768, 
                 encoder_depth=12,
                 encoder_num_heads=12, 
                 decoder_num_classes=256, 
                 decoder_embed_dim=512, 
                 decoder_depth=8,
                 decoder_num_heads=8, 
                 mlp_ratio=4., 
                 qkv_bias=False, 
                 qk_scale=None, 
                 drop_rate=0., 
                 attn_drop_rate=0.,
                 drop_path_rate=0., 
                 norm_layer=nn.LayerNorm, 
                 init_values=0.,
                 use_learnable_pos_emb=False,
                 use_checkpoint=False,
                 tubelet_size=2,
                 n_parts=20,
                 mask_ratio=0.5,
                 target_type='pixel',
                 mask_type='tube',
                 memory_size=0,
                 loss_func='SWAV',
                 num_prototypes=3000,
                 world_size=0,
                 sinkhorn_iterations=10,
                 eps=0.05,
                 kwindow=1,
                 skip_cls_dino=False,
                 num_classes=0, # avoid the error from create_fn in timm
                 in_chans=0, # avoid the error from create_fn in timm
                 pretrained_cfg=None, # avoid the error from create_fn in timm
                 pretrained_cfg_overlay=None,
                 num_frames=None,  # Added num_frames parameter with default None
                 dino_out_dim=65536,  # number of dimensions in DINO head
                 dino_hidden_dim=2048,  # hidden dimensions in DINO head
                 dino_bottleneck_dim=256,  # bottleneck dimensions
                 dino_warmup_teacher_temp=0.04,  # Default initial teacher temperature
                 dino_teacher_temp=0.07,  # Default final teacher temperature
                 dino_warmup_teacher_temp_epochs=30,  # Default warmup period
                 momentum_teacher=0.99,  # Default teacher momentum
                 center_momentum=0.8,  # Default center momentum
                 tokenizer_type='default',
                 skip_dino_loss=False,  # Added center momentum parameter
                 use_turbo_training=False,
                 turbo_recon_ratio=0.25,
                 use_mean_pooling=False,
                 fc_drop_rate=0.,
                 init_scale=0.,
                 dino_only=False,
                 use_combined_dino_swav=False,
                 use_video_dino=False,
                 use_dino_crop=False,
                 local_size=96,
                 num_local_views=1,
                 grayscale_mode=False):  # Added grayscale_mode parameter
        super().__init__()
        # print(f"PretrainVisionTransformer init - local_size: {local_size}")
        self.dino_only = dino_only
        self.use_combined_dino_swav = use_combined_dino_swav
        self.use_video_dino = use_video_dino
        self.normalize_features = True
        self.img_size = img_size
        self.skip_dino_loss = skip_dino_loss
        self.use_dino_crop = use_dino_crop
        self.local_size = local_size
        self.num_local_views = num_local_views
        self.patch_size = patch_size
        self.grayscale_mode = grayscale_mode  # Store grayscale mode
        
        # Modify input channels based on grayscale mode
        if grayscale_mode:
            encoder_in_chans = 1  # Change to 1 channel for grayscale
        
        # print('Image_size',self.img_size)
        if use_combined_dino_swav:
            # Create image student only (using same VIT architecture)
            self.register_buffer("image_center", torch.zeros(1, dino_out_dim))
            self.register_buffer("video_center", torch.zeros(1, dino_out_dim))
            self.image_student = PretrainVisionTransformerEncoder(
                img_size=self.img_size,
                patch_size=patch_size,
                in_chans=encoder_in_chans,
                embed_dim=encoder_embed_dim,
                depth=encoder_depth,
                num_heads=encoder_num_heads,
                mlp_ratio=mlp_ratio,
                tokenizer_type='default',
                init_values=init_values,
                tubelet_size=1,
                num_frames=1,
                use_learnable_pos_emb=False,
                use_mean_pooling=False,
                use_checkpoint=False,
                skip_dino_loss=self.skip_dino_loss,
                use_dino_crop=self.use_dino_crop,
                local_size=self.local_size,
            )
            # Create teacher with same image size
            self.image_teacher = copy.deepcopy(self.image_student)
            for param in self.image_teacher.parameters():
                 param.requires_grad = False
            
            # Create video teacher only if video_dino is enabled
            

        # Pass tokenizer_type to the encoder
        self.encoder = PretrainVisionTransformerEncoder(
            img_size=self.img_size, 
            patch_size=patch_size, 
            in_chans=encoder_in_chans, 
            num_classes=encoder_num_classes, 
            embed_dim=encoder_embed_dim, 
            depth=encoder_depth,
            num_heads=encoder_num_heads, 
            mlp_ratio=mlp_ratio, 
            qkv_bias=qkv_bias, 
            qk_scale=qk_scale, 
            drop_rate=drop_rate, 
            attn_drop_rate=attn_drop_rate,
            drop_path_rate=drop_path_rate, 
            norm_layer=norm_layer, 
            init_values=init_values,
            tubelet_size=tubelet_size,
            use_checkpoint=use_checkpoint,
            mask_type=mask_type,
            use_learnable_pos_emb=use_learnable_pos_emb, 
            num_frames=num_frames,
            tokenizer_type=tokenizer_type,
            skip_dino_loss=skip_dino_loss,
            use_dino_crop=use_dino_crop,
            local_size=self.local_size
        )
        if self.use_video_dino:
                self.video_teacher = copy.deepcopy(self.encoder)
                for param in self.video_teacher.parameters():
                    param.requires_grad = False
                
            
        # Calculate num_patches based on tokenizer_type
        if tokenizer_type == 'default':
            # print('DEFAULT_MEIN_AAYA')
            # Original patch-based calculation
            self.patch_size = patch_size
            self.tubelet_size = tubelet_size
            self.num_frames = num_frames
            # This is equal to 14*14 = 196 for standard VideoMAE with 224 resolution
            num_patches = (img_size // patch_size) * (img_size // patch_size) * (num_frames // tubelet_size)
        elif tokenizer_type == 'sparse_tube':  # sparse_tube
            # print('SPARSE_TUBE_MEIN_AAYA')
            # For sparse tube tokenizer, get the number from encoder
            num_patches = self.encoder.patch_embed.num_patches 
            
        # Only create decoder if not dino_only
        if not dino_only:
            self.decoder = PretrainVisionTransformerDecoder(
                patch_size=patch_size,
                num_patches=num_patches,
                num_classes=decoder_num_classes,
                embed_dim=decoder_embed_dim,
                depth=decoder_depth,
                num_heads=decoder_num_heads, 
                mlp_ratio=mlp_ratio, 
                qkv_bias=qkv_bias, 
                qk_scale=qk_scale, 
                drop_rate=drop_rate, 
                attn_drop_rate=attn_drop_rate,
                drop_path_rate=drop_path_rate, 
                norm_layer=norm_layer, 
                init_values=init_values,
                tubelet_size=tubelet_size,
                use_checkpoint=use_checkpoint,
                num_frames=num_frames
            )
            self.mask_token = nn.Parameter(torch.zeros(1, 1, decoder_embed_dim))
            trunc_normal_(self.mask_token, std=.02)

            self.encoder_to_decoder = nn.Linear(encoder_embed_dim, decoder_embed_dim, bias=False)
        self.world_size = world_size
        self.eps = eps
        self.sinkhorn_iterations=sinkhorn_iterations
        
        # Initialize position embeddings
        num_patches = self.encoder.patch_embed.num_patches
        if not self.skip_dino_loss:  # If using DINO at all (either standard or combined)
            # Add +1 for class token position
            self.pos_embed = get_sinusoid_encoding_table(num_patches + 1, decoder_embed_dim)
            print('STANDARD_POS_EMBED_INSIDE_MAIN',self.pos_embed.shape)
        else:
            # Only skip CLS token if we're completely skipping DINO
            self.pos_embed = get_sinusoid_encoding_table(num_patches, decoder_embed_dim)

        self.mask_ratio = mask_ratio
        self.target_type = target_type
        self.mask_type = mask_type
        self.decoder_num_classes = decoder_num_classes
        
        self.memory_size = memory_size
        if memory_size > 0:
            self.memory = torch.nn.Parameter(torch.randn(memory_size, decoder_num_classes))
            self.memory_pred_head = torch.nn.Sequential(
                                        torch.nn.Linear(decoder_num_classes, decoder_num_classes),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(decoder_num_classes, decoder_num_classes),
                                        torch.nn.GELU(),
                                        torch.nn.Linear(decoder_num_classes, memory_size)
                                    )
        
        def init_proj_mlp(input_dim):
            # return nn.Linear(input_dim, decoder_num_classes)
        
            return torch.nn.Sequential(
                    torch.nn.Linear(input_dim, 1024),
                    torch.nn.GELU(),
                    torch.nn.Linear(1024, 1024),
                    torch.nn.GELU(),
                    torch.nn.Linear(1024, 512),
                    torch.nn.GELU(),
                    torch.nn.Linear(512, decoder_num_classes),
            )
            
        self.loss_func = loss_func
        if loss_func == 'SWAV' and not dino_only:
            self.num_prototypes = num_prototypes
            self.prototypes = torch.nn.Parameter(torch.randn(num_prototypes, decoder_num_classes))

        if ('dino' in target_type or self.use_combined_dino_swav or self.dino_only) and self.loss_func == 'SWAV':
            self.head = nn.Identity()
        elif target_type == 'mlp':
            self.head = init_proj_mlp(1536)
                
        self.skip_dino_loss = skip_dino_loss
            
        # DINO specific
        self.dino_out_dim = dino_out_dim
        self.dino_warmup_teacher_temp = dino_warmup_teacher_temp
        self.dino_teacher_temp = dino_teacher_temp
        self.dino_warmup_teacher_temp_epochs = dino_warmup_teacher_temp_epochs
        self.momentum_teacher = momentum_teacher
        self.center_momentum = center_momentum
        self.student_temp = 0.1  # Default student temperature

        # Pre-compute teacher temperature schedule
        self.teacher_temp_schedule = torch.tensor(np.concatenate((
            np.linspace(dino_warmup_teacher_temp, dino_teacher_temp, dino_warmup_teacher_temp_epochs),
            np.ones(400 - dino_warmup_teacher_temp_epochs) * dino_teacher_temp  # Assuming 800 epochs total
        )), dtype=torch.float32)

        # Add cross-modal attention
        # if self.use_combined_dino_swav and self.use_video_dino:
        #     self.cross_modal_attention = CrossModalAttention(
        #         dim=encoder_embed_dim,
        #         num_heads=encoder_num_heads,
        #         dropout=drop_rate
        #     )

        # Create student DINO heads
        if not self.skip_dino_loss:
            if self.use_combined_dino_swav:
                # Image DINO heads
                self.image_dino_head = DINOHead(
                    encoder_embed_dim,
                    dino_out_dim,
                    hidden_dim=dino_hidden_dim,
                    bottleneck_dim=dino_bottleneck_dim
                )
                self.image_teacher_head = DINOHead(
                    encoder_embed_dim,
                    dino_out_dim,
                    hidden_dim=dino_hidden_dim,
                    bottleneck_dim=dino_bottleneck_dim
                )
                # Disable gradients for image teacher head
                for p in self.image_teacher_head.parameters():
                    p.requires_grad = False
            
                # Video DINO heads (if video DINO is enabled)
                if self.use_video_dino:
                    self.video_dino_head = DINOHead(
                        encoder_embed_dim,
                        dino_out_dim,
                        hidden_dim=dino_hidden_dim,
                        bottleneck_dim=dino_bottleneck_dim
                    )
                    self.video_teacher_head = DINOHead(
                        encoder_embed_dim,
                        dino_out_dim,
                        hidden_dim=dino_hidden_dim,
                        bottleneck_dim=dino_bottleneck_dim
                    )
                    # Disable gradients for video teacher head
                    for p in self.video_teacher_head.parameters():
                        p.requires_grad = False

                # Initialize image teacher with student weights
                self.update_teacher(momentum=0, update_image=True, update_video=False)
                
                # Initialize video teacher if enabled
                if self.use_video_dino:
                    self.update_teacher(momentum=0, update_image=False, update_video=True)
            else:
                # Standard DINO mode
                self.dino_head = DINOHead(
                    encoder_embed_dim,
                    dino_out_dim,
                    hidden_dim=dino_hidden_dim,
                    bottleneck_dim=dino_bottleneck_dim
                )
                self.teacher = create_teacher_model(self)
                self.teacher_head = DINOHead(
                    encoder_embed_dim,
                    dino_out_dim,
                    hidden_dim=dino_hidden_dim,
                    bottleneck_dim=dino_bottleneck_dim
                )
                
                # Disable gradient updates for teacher
                for p in self.teacher.parameters():
                    p.requires_grad = False
                for p in self.teacher_head.parameters():
                    p.requires_grad = False

                # Initialize teacher with student weights
                self.update_teacher(momentum=0)  # First update with momentum=0 to copy weights

                # DINO center register
                self.register_buffer("center", torch.zeros(1, dino_out_dim))
                self.center_momentum = center_momentum
                self.student_temp = 0.1  # DINO default student temperature

        self.use_turbo_training = use_turbo_training
        self.turbo_recon_ratio = turbo_recon_ratio
        self.use_dino_crop = use_dino_crop

    def extract_assignments(self, projected_features, detach=False):
        bs, np, dim = projected_features.shape
        projected_dim = projected_features.shape[-1]
        projected_features = projected_features.reshape(-1, projected_dim)
        normalized_projected_features = F.normalize(projected_features, dim=-1, p=2)
        
        prototypes = self.prototypes.detach() if detach else self.prototypes
            
        batch_scores = torch.einsum('bd,nd->bn', normalized_projected_features, prototypes)
        batch_q = utils.find_optimal_assignment(batch_scores, self.eps, self.sinkhorn_iterations, world_size=self.world_size)
        batch_q = batch_q.reshape(bs, np, self.num_prototypes)
        batch_scores = batch_scores.reshape(bs, np, self.num_prototypes)
        return batch_scores,batch_q

    def process_image_dino_batch(self, imgs_batch, epoch):
        """
        Process a batch of frames for image DINO and return loss
        Args:
            imgs_batch: Tensor of shape [batch_size, C, H, W, 1]
            epoch: Current epoch for temperature
        """
        # Get correct number of patches from image teacher model
        num_patches = self.image_teacher.patch_embed.num_patches
        batch_size = imgs_batch.shape[0]
        num_masked = int(num_patches * self.mask_ratio)
        
        # Generate masks based on mask_type
        if self.mask_type == 'tube':
            rand = torch.rand(batch_size, num_patches, device=imgs_batch.device)
            image_masks = rand.topk(k=num_masked, dim=1).indices
            image_masks_binary = torch.zeros(batch_size, num_patches, device=imgs_batch.device, dtype=torch.bool)
            image_masks_binary.scatter_(1, image_masks, True)  # [B, 196]
        elif self.mask_type == 'multi_local':
            rand = torch.rand(batch_size, self.num_local_views, num_patches, device=imgs_batch.device)
            local_masks = rand.topk(k=num_masked, dim=-1).indices
            local_masks_binary = torch.zeros(batch_size, self.num_local_views, num_patches, device=imgs_batch.device, dtype=torch.bool)
            local_masks_binary.scatter_(-1, local_masks, True)  # [B, N, 196]
        
        # Teacher path
        with torch.no_grad():
            teacher_frame_features, _ = self.image_teacher(
                imgs_batch, 
                torch.zeros_like(image_masks_binary if self.mask_type == 'tube' else local_masks_binary[:, 0]),
                is_dino=True
            )
            teacher_cls = teacher_frame_features[:, 0]
            teacher_output = self.image_teacher_head(teacher_cls)
            
            # Debug: Print teacher output stats
            # print(f"\nTeacher Output Stats:")
            # print(f"Min: {teacher_output.min().item():.4f}")
            # print(f"Max: {teacher_output.max().item():.4f}")
            # print(f"Mean: {teacher_output.mean().item():.4f}")
            # print(f"Std: {teacher_output.std().item():.4f}")
            
            # Ensure teacher temperature doesn't get too small
            current_teacher_temp = max(0.04, self.get_dino_temp(epoch, is_teacher=True))
            # print(f"Teacher Temperature: {current_teacher_temp:.4f}")
            
            teacher_output = (teacher_output - self.image_center) / current_teacher_temp
            teacher_output = F.softmax(teacher_output, dim=-1)
            
            # Debug: Print teacher output after softmax
            # print(f"\nTeacher Output After Softmax:")
            # print(f"Min: {teacher_output.min().item():.4f}")
            # print(f"Max: {teacher_output.max().item():.4f}")
            # print(f"Mean: {teacher_output.mean().item():.4f}")
            # print(f"Std: {teacher_output.std().item():.4f}")

        # Student path
        student_outputs = []
        # Ensure student temperature stays in a reasonable range
        current_student_temp = max(0.1, min(0.2, self.get_dino_temp(epoch, is_teacher=False)))
        # print(f"Student Temperature: {current_student_temp:.4f}")
        
        # Local views
        if self.mask_type == 'multi_local':
            for local_mask in local_masks_binary.transpose(0, 1):  # [N, B, 196]
                local_student_features, _ = self.image_student(
                    imgs_batch, 
                    local_mask,
                    is_dino=True
                )
                local_student_cls = local_student_features[:, 0]
                local_student_output = self.image_dino_head(local_student_cls)
                student_outputs.append(local_student_output)
        else:
            # Single masked view
            local_student_features, _ = self.image_student(
                imgs_batch, 
                image_masks_binary,
                is_dino=True
            )
            local_student_cls = local_student_features[:, 0]
            local_student_output = self.image_dino_head(local_student_cls)
            student_outputs.append(local_student_output)
        
        # Debug: Print student outputs
        # for i, student_out in enumerate(student_outputs):
        #     print(f"\nStudent Output {i} Stats:")
        #     print(f"Min: {student_out.min().item():.4f}")
        #     print(f"Max: {student_out.max().item():.4f}")
        #     print(f"Mean: {student_out.mean().item():.4f}")
        #     print(f"Std: {student_out.std().item():.4f}")
        
        # Compute DINO loss with gradient scaling
        batch_dino_loss = 0.0
        for student_out in student_outputs:
            student_logsoftmax = F.log_softmax(student_out / current_student_temp, dim=-1)
            batch_dino_loss += -torch.mean(torch.sum(teacher_output * student_logsoftmax, dim=-1))
        
        # Scale loss based on batch size
        grad_scale = 256.0 / imgs_batch.shape[0]  # Reference batch size of 256
        batch_dino_loss = batch_dino_loss * grad_scale
        
        # Normalize loss by number of views
        batch_dino_loss = batch_dino_loss / len(student_outputs)
        
        # Update center AFTER loss computation
        self.update_center(teacher_output, is_video=False)
        
        # Debug prints
        # if torch.isnan(batch_dino_loss) or torch.isinf(batch_dino_loss):
        #     print(f"Warning: Invalid loss value detected - {batch_dino_loss}")
        #     print(f"Teacher temp: {current_teacher_temp}, Student temp: {current_student_temp}")
        #     print(f"Teacher output stats - min: {teacher_output.min()}, max: {teacher_output.max()}, mean: {teacher_output.mean()}")
        #     for i, student_out in enumerate(student_outputs):
        #         print(f"Student {i} output stats - min: {student_out.min()}, max: {student_out.max()}, mean: {student_out.mean()}")
        
        return batch_dino_loss

    def normalize_prototypes(self):
        with torch.no_grad():
            w = self.prototypes.data.clone()
            w = F.normalize(w, dim=1, p=2)
            self.prototypes.copy_(w)
            
    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.xavier_uniform_(m.weight)
            if isinstance(m, nn.Linear) and m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def get_num_layers(self):
        return len(self.blocks)

    @torch.jit.ignore
    def no_weight_decay(self):
        return {'pos_embed', 'cls_token', 'mask_token'}

    @torch.no_grad()
    def update_teacher(self, momentum=None, update_video=True, update_image=True):
        """Update teacher networks with momentum
        Args:
            momentum: Optional override for momentum value
            update_video: Whether to update video teacher and head
            update_image: Whether to update image teacher and head
        """
        m = momentum if momentum is not None else self.momentum_teacher
        
        if self.use_combined_dino_swav:
            # Update image components
            if update_image:
                # Update image encoder
                for param_student, param_teacher in zip(self.image_student.parameters(), 
                                                      self.image_teacher.parameters()):
                    param_teacher.data.mul_(m).add_(param_student.data, alpha=1.0 - m)
                # Update image head
                for param_student, param_teacher in zip(self.image_dino_head.parameters(), 
                                                      self.image_teacher_head.parameters()):
                    param_teacher.data.mul_(m).add_(param_student.data, alpha=1.0 - m)
            
            # Update video components
            if self.use_video_dino and update_video:
                # Update video encoder
                for param_student, param_teacher in zip(self.encoder.parameters(), 
                                                      self.video_teacher.parameters()):
                    param_teacher.data.mul_(m).add_(param_student.data, alpha=1.0 - m)
                # Update video head
                for param_student, param_teacher in zip(self.video_dino_head.parameters(), 
                                                      self.video_teacher_head.parameters()):
                    param_teacher.data.mul_(m).add_(param_student.data, alpha=1.0 - m)
        else:
            # Standard DINO mode (not combined with SWAV)
            for param_student, param_teacher in zip(self.encoder.parameters(), self.teacher.parameters()):
                param_teacher.data.mul_(m).add_(param_student.data, alpha=1.0 - m)
            for param_student, param_teacher in zip(self.dino_head.parameters(), self.teacher_head.parameters()):
                param_teacher.data.mul_(m).add_(param_student.data, alpha=1.0 - m)
    @torch.no_grad()
    def update_center(self, teacher_output, is_video=False):
        """Update center for teacher output
        Args:
            teacher_output: Teacher model output
            is_video: Whether this is for video path
        """
        batch_center = torch.sum(teacher_output, dim=0, keepdim=True)
        if dist.is_initialized():
            dist.all_reduce(batch_center)
            batch_center = batch_center / (len(teacher_output) * dist.get_world_size())
        else:
            batch_center = batch_center / len(teacher_output)
        
        if self.use_combined_dino_swav:
            if is_video:
                self.video_center = self.video_center * self.center_momentum + batch_center * (1 - self.center_momentum)
            else:
                self.image_center = self.image_center * self.center_momentum + batch_center * (1 - self.center_momentum)
        else:
            self.center = self.center * self.center_momentum + batch_center * (1 - self.center_momentum)

    def get_dino_temp(self, epoch, is_teacher=True):
        if is_teacher:
            return self.teacher_temp_schedule[epoch]
        else:
            return self.student_temp

    def extract_swav_teacher_features(self, imgs):
        """
        Extract features from the image teacher model for SWAV training
        Args:
            imgs: Tensor of shape [batch_size, C, H, W, 1]
        Returns:
            features: Tensor of shape [batch_size, num_patches, embed_dim]
        """
        with torch.no_grad():
            # Process through image teacher
            features, _ = self.image_teacher(
                imgs,
                torch.zeros_like(torch.zeros(imgs.shape[0], self.image_teacher.patch_embed.num_patches, device=imgs.device, dtype=torch.bool)),
                is_dino=True
            )
            
            # Remove CLS token and reshape if needed
            if not self.skip_dino_loss:
                features = features[:, 1:]  # Remove CLS token
            
            # Normalize features if needed
            if self.normalize_features:
                features = (features - features.mean(dim=-2, keepdim=True)) / (
                    features.var(dim=-2, unbiased=True, keepdim=True).sqrt() + 1e-6)
            
            return features

    def forward(self, imgs, masks, labels=None, epoch=None, turbo_mask=None, local_videos=None):
        if self.use_combined_dino_swav:
            B, C, T, H_img, W_img = imgs.shape
            
            # Update teachers - can be controlled separately
            self.update_teacher(update_image=True, update_video=self.use_video_dino)
            
            # Image DINO path
            imgs_reshaped = imgs.transpose(1, 2).reshape(-1, C, H_img, W_img)
            imgs_reshaped = imgs_reshaped.unsqueeze(2)
            
            batch_size = 1024 # Adjust based on memory constraints
            total_frames = imgs_reshaped.shape[0]
            dino_loss = torch.tensor(0.0, device=imgs.device)
    
            for start_idx in range(0, total_frames, batch_size):
                end_idx = min(start_idx + batch_size, total_frames)
                batch_frames = imgs_reshaped[start_idx:end_idx]
                
                image_dino_loss = self.process_image_dino_batch(
                    imgs_batch=batch_frames,
                    epoch=epoch
                )
                
            
            # Extract SWAV teacher features after image DINO training
            swav_teacher_features = self.extract_swav_teacher_features(imgs_reshaped)
            swav_teacher_features = swav_teacher_features.reshape(B, T, -1, swav_teacher_features.shape[-1])
            swav_teacher_features = rearrange(swav_teacher_features, 'b t n c -> b (t n) c')
            swav_teacher_features = swav_teacher_features[:,::2,:]  # Take every other frame
            
            if self.mask_type == 'multi_local':
                masks, local_masks = masks
            else:
                local_masks = None
            
            enc_out_global, _ = self.encoder(imgs, masks)
            
            # Video DINO path (only if enabled)
            if self.use_video_dino:
                # Teacher path with correct temperature
                with torch.no_grad():
                    video_teacher_out, _ = self.video_teacher(imgs, torch.zeros_like(masks))
                    video_teacher_cls = video_teacher_out[:, 0]
                    video_teacher_output = self.video_teacher_head(video_teacher_cls)
                    current_teacher_temp = self.get_dino_temp(epoch, is_teacher=True)
                    video_teacher_output = (video_teacher_output - self.video_center) / current_teacher_temp
                    video_teacher_output = F.softmax(video_teacher_output, dim=-1)
                
                local_student_outputs = []
                
                # Handle local views
                if hasattr(self, 'use_dino_crop') and self.use_dino_crop and local_videos is not None:
                    # Store original pos embed
                    orig_pos_embed_video = self.encoder.pos_embed
                    local_patch_size = self.local_size // self.patch_size
                    
                    local_mask = torch.zeros(B, (T//2)*local_patch_size * local_patch_size, device=imgs.device, dtype=torch.bool)
                    # Switch to local pos embed for local views
                    self.encoder.pos_embed = self.encoder.local_pos_embed_video
                    orig_img_size = self.encoder.patch_embed.img_size
                    self.encoder.patch_embed.img_size = (self.local_size, self.local_size)
                    local_videos = local_videos.permute(1,0,2,3,4,5)
                    for local_view in local_videos:
                        enc_out_local, _ = self.encoder(local_view, local_mask)
                        local_cls = enc_out_local[:, 0]
                        local_student_out = self.video_dino_head(local_cls)
                        local_student_outputs.append(local_student_out)
                    self.encoder.patch_embed.img_size = orig_img_size
                    self.encoder.pos_embed = orig_pos_embed_video
                elif self.mask_type == 'multi_local' and local_masks is not None:
                    for local_mask in local_masks:
                        enc_out_local, _ = self.encoder(imgs, local_mask)
                        local_cls = enc_out_local[:, 0]
                        local_student_out = self.video_dino_head(local_cls)
                        local_student_outputs.append(local_student_out)
                
                # Compute video DINO loss with proper normalization and scaling
                video_dino_loss = torch.tensor(0.0, device=imgs.device)
                current_student_temp = self.get_dino_temp(epoch, is_teacher=False)
                for student_output in local_student_outputs:
                    video_student_logsoftmax = F.log_softmax(student_output / current_student_temp, dim=-1)
                    video_dino_loss += -torch.mean(torch.sum(video_teacher_output * video_student_logsoftmax, dim=-1))
                
                # Normalize by number of views
                video_dino_loss = video_dino_loss / len(local_student_outputs)
                
                # Update center after loss computation
                self.update_center(video_teacher_output, is_video=True)
            
            # SWAV path with teacher features
            if not self.skip_dino_loss:
                x_vis = self.encoder_to_decoder(enc_out_global[:, 1:])  # Remove CLS token
            else:
                x_vis = self.encoder_to_decoder(enc_out_global)  # No CLS token to remove
            
            B_vis, N, C = x_vis.shape
            
            # Handle position embeddings based on presence of class token
            if not self.skip_dino_loss:
                expand_pos_embed = self.pos_embed[:, 1:].expand(B_vis, -1, -1).clone().detach().to(imgs.device).type_as(imgs)
            else:
                expand_pos_embed = self.pos_embed.expand(B_vis, -1, -1).clone().detach().to(imgs.device).type_as(imgs)
                
            # Split into visible and masked tokens using the mask
            pos_emd_vis = expand_pos_embed[~masks].reshape(B_vis, -1, C)
            pos_emd_mask = expand_pos_embed[masks].reshape(B_vis, -1, C)

            # Add position embeddings and concatenate with mask tokens
            x_full = torch.cat([
                x_vis + pos_emd_vis,  # visible tokens with position embeddings
                self.mask_token + pos_emd_mask  # masked tokens with position embeddings
            ], dim=1)
            # print('X_FULL',x_full.shape)
            # Get number of tokens to decode
            return_token = pos_emd_mask.shape[1]  # number of masked tokens
            x = self.decoder(x_full, 0)
            # print('X',x.shape)
            
            if self.loss_func == 'SWAV':
                self.normalize_prototypes()
                scores1, q1 = self.extract_assignments(swav_teacher_features, detach=True)
                scores2, q2 = self.extract_assignments(x, detach=False)
            
            if self.use_video_dino:
                return x, (scores1, q1), (scores2, q2, video_dino_loss, image_dino_loss)
            else:
                return x, (scores1, q1), (scores2, q2, image_dino_loss)
        
        else:
            # Handle masks based on mask_type
            if self.mask_type == 'tube':
                # For tube masking, use the first mask (global)
                if isinstance(masks, tuple):
                    global_mask = masks[0]
                else:
                    global_mask = masks
                local_masks = None
            elif self.mask_type == 'multi_local':
                # For multi_local, use both global and local masks
                if isinstance(masks, tuple) and len(masks) == 2:
                    global_mask, local_masks = masks
                else:
                    raise ValueError("multi_local mask type requires tuple of (global_mask, local_masks)")
            elif self.mask_type in ['random', 'parts', 'tube_fgbg']:
                # For other mask types, use single mask
                if isinstance(masks, tuple):
                    global_mask = masks[0]
                else:
                    global_mask = masks
                local_masks = None
            else:
                raise ValueError(f"Unsupported mask type: {self.mask_type}")
            
            B, C, T, H_img, W_img = imgs.shape  # imgs: [B, C, T, H_img, W_img]
            enc_out_global, mask_pos = self.encoder(imgs, global_mask)
            if self.skip_dino_loss:
                dino_loss = torch.tensor(0.0, device=imgs.device)
            else:
                # -------- Update Teacher --------
                self.update_teacher()

                # -------- Student Branch: Global View --------
                # cls_token = enc_out_global[:, 0]  
                # student_output_global = self.dino_head(cls_token)
                
                # -------- Teacher Branch (Global View) --------
                with torch.no_grad():
                    teacher_out, _ = self.teacher(imgs, torch.zeros_like(global_mask))
                    teacher_cls = teacher_out[:, 0]
                    teacher_output = self.teacher_head(teacher_cls)
                    current_teacher_temp = self.get_dino_temp(epoch, is_teacher=True)
                    teacher_output = (teacher_output - self.center) / current_teacher_temp
                    teacher_output = F.softmax(teacher_output, dim=-1)
                    

                # -------- Student Branch: Local Views --------
                # student_outputs = [student_output_global]
                student_outputs = []
                if local_masks is not None:
                    for local_mask in local_masks:
                        enc_out_local, _ = self.encoder(imgs, local_mask)
                        local_cls = enc_out_local[:, 0]
                        local_student_output = self.dino_head(local_cls)
                        student_outputs.append(local_student_output)

            # -------- Compute DINO Loss --------
                dino_loss = 0.
                current_student_temp = self.get_dino_temp(epoch, is_teacher=False)
                for student_logits in student_outputs:
                    student_logsoftmax = F.log_softmax(student_logits / current_student_temp, dim=-1)
                    dino_loss += -torch.mean(torch.sum(teacher_output * student_logsoftmax, dim=-1))
                
                # Normalize loss by number of views
                dino_loss = dino_loss / len(student_outputs)
                self.update_center(teacher_output, is_video=False)
        # Add check for dino_only mode
        if self.dino_only:
            return dino_loss
        
        # -------- SIGMA (Reconstruction) Branch --------
        # Remove the CLS token for reconstruction (using only patch tokens)
        if not self.skip_dino_loss:
            x_vis = self.encoder_to_decoder(enc_out_global[:, 1:])  # shape: [B, N, C]
        else:
            x_vis = self.encoder_to_decoder(enc_out_global)  # shape: [B, N, C]
        B_vis, N, C = x_vis.shape
        
        # Determine which mask to use for reconstruction
        if self.use_turbo_training and turbo_mask is not None:
            # Use the provided turbo mask for reconstruction
            if isinstance(turbo_mask, tuple):
                recon_mask = turbo_mask[0]
            else:
                recon_mask = turbo_mask
        else:
            # Use original mask for standard training
            recon_mask = global_mask
        
        # Handle position embeddings based on presence of class token
        if not self.skip_dino_loss:
            # When we have class token, skip its position embedding
            expand_pos_embed = self.pos_embed[:, 1:].expand(B_vis, -1, -1).clone().detach().to(imgs.device).type_as(imgs)
        else:
            # When no class token, use full position embeddings
            expand_pos_embed = self.pos_embed.expand(B_vis, -1, -1).clone().detach().to(imgs.device).type_as(imgs)

        if self.mask_type in ['tube', 'tube_fgbg', 'parts', 'multi_local']:
            pos_emd_vis = expand_pos_embed[~global_mask].reshape(B_vis, -1, C)
        pos_emd_mask = expand_pos_embed[recon_mask].reshape(B_vis, -1, C)
        
        return_token = pos_emd_mask.shape[1]
        x_full = torch.cat([x_vis + pos_emd_vis, self.mask_token + pos_emd_mask], dim=1)
        x = self.decoder(x_full, return_token)

        # Memory processing
        if self.memory_size > 0:
            mem_query = self.memory_pred_head(x)
            softmax = F.softmax(mem_query, dim=-1)
            x = torch.einsum('bnm,lmd->bnld', softmax, self.memory.unsqueeze(0))
            x = x.squeeze(2)

        # Process SWAV assignments (for both standard and Turbo Training)
        scores1, q1 = None, None
        scores2, q2 = None, None
        
        if 'mlp' in self.target_type or (('dino' in self.target_type) and self.loss_func == 'SWAV'):
            proj = self.head(labels)

        if 'mha' in self.target_type:
            proj = proj.reshape(B_vis, T//2, 196, self.decoder_num_classes).flatten(0, 1)
            proj = self.mha(self.key(proj), self.value(proj), self.query(proj),
                            attn_mask=self.attn_mask.to(proj.device))[0]
            proj = proj.reshape(B_vis, T//2, 196, self.decoder_num_classes).flatten(1, 2)
            proj = self.head2(proj)
            proj = proj[global_mask].reshape(B_vis, -1, self.decoder_num_classes)

        # Process SWAV loss (for both standard and Turbo Training)
        if self.loss_func == 'SWAV':
            self.normalize_prototypes()
            scores1, q1 = self.extract_assignments(proj, detach=False)
            scores2, q2 = self.extract_assignments(x, detach=True)
        
        # Return the same structure for both standard and Turbo Training
        return x, (scores1, q1), (scores2, q2, dino_loss)
        

def create_teacher_model(student):
    """Create a teacher model with same architecture as student"""
    
    teacher = copy.deepcopy(student.encoder)
    return teacher

@register_model
def pretrain_videomae_small_patch16_224(pretrained=False, **kwargs):
    input_size = kwargs.pop('input_size', 224)
    grayscale_mode = kwargs.pop('grayscale_mode', False)
    model = PretrainVisionTransformer(
        patch_size=16,
        encoder_embed_dim=384,
        encoder_depth=12,
        encoder_num_heads=6,
        decoder_embed_dim=192, 
        decoder_num_heads=3,
        mlp_ratio=4,
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6),
        img_size=input_size,
        grayscale_mode=grayscale_mode,
        **kwargs,
    )
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_videomae_base_patch16_224(pretrained=False, **kwargs):
    grayscale_mode = kwargs.pop('grayscale_mode', False)
    model = PretrainVisionTransformer(
        patch_size=16, 
        encoder_embed_dim=768, 
        encoder_depth=12, 
        encoder_num_heads=12,
        encoder_num_classes=0,
        decoder_embed_dim=384,
        decoder_num_heads=6,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        grayscale_mode=grayscale_mode,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
 
@register_model
def pretrain_videomae_large_patch16_224(pretrained=False, **kwargs):
    input_size = kwargs.pop('input_size', 224)
    grayscale_mode = kwargs.pop('grayscale_mode', False)
    model = PretrainVisionTransformer(
        patch_size=16, 
        encoder_embed_dim=1024, 
        encoder_depth=24, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_embed_dim=512,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        img_size=input_size,
        grayscale_mode=grayscale_mode,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model

@register_model
def pretrain_videomae_huge_patch16_224(pretrained=False, **kwargs):
    input_size = kwargs.pop('input_size', 224)
    grayscale_mode = kwargs.pop('grayscale_mode', False)
    model = PretrainVisionTransformer(
        patch_size=16, 
        encoder_embed_dim=1280, 
        encoder_depth=32, 
        encoder_num_heads=16,
        encoder_num_classes=0,
        decoder_embed_dim=640,
        decoder_num_heads=8,
        mlp_ratio=4, 
        qkv_bias=True,
        norm_layer=partial(nn.LayerNorm, eps=1e-6), 
        img_size=input_size,
        grayscale_mode=grayscale_mode,
        **kwargs)
    model.default_cfg = _cfg()
    if pretrained:
        checkpoint = torch.load(
            kwargs["init_ckpt"], map_location="cpu"
        )
        model.load_state_dict(checkpoint["model"])
    return model
