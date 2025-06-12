import numpy as np
import torch
from discovr.models.TubeViT.tubevit.model import SparseTubesTokenizer

class TubeMaskingGenerator:
    def __init__(self, input_size, mask_ratio):
        self.frames, self.height, self.width = input_size
        self.num_patches_per_frame =  self.height * self.width
        self.total_patches = self.frames * self.num_patches_per_frame 
        self.num_masks_per_frame = int(mask_ratio * self.num_patches_per_frame)
        self.total_masks = self.frames * self.num_masks_per_frame

    def __repr__(self):
        repr_str = "Maks: total patches {}, mask patches {}".format(
            self.total_patches, self.total_masks
        )
        return repr_str

    def __call__(self):
        mask_per_frame = np.hstack([
            np.zeros(self.num_patches_per_frame - self.num_masks_per_frame),
            np.ones(self.num_masks_per_frame),
        ])
        np.random.shuffle(mask_per_frame)
        mask = np.tile(mask_per_frame, (self.frames,1)).flatten()
        
        # Convert to PyTorch boolean tensor
        mask = torch.from_numpy(mask).bool()
        
        return mask, []  # Now returns a PyTorch boolean tensor

class MultiLocalMaskingGenerator:
    def __init__(self, window_size, global_mask_ratio, local_mask_ratio, num_local_views, 
                 masking_mode='random', visible_ratio=0.4):
        """Initialize masking generator"""
        print(f"DEBUG: Initializing MultiLocalMaskingGenerator with window_size={window_size}")
        self.T, self.H, self.W = window_size  # Take all three dimensions (T,H,W)
        self.num_patches_per_frame = self.H * self.W
        self.total_patches = self.T * self.num_patches_per_frame  # Total patches across all frames
        self.global_mask_ratio = global_mask_ratio
        self.local_mask_ratio = local_mask_ratio
        self.visible_ratio = visible_ratio
        self.num_local_views = num_local_views
        self.masking_mode = masking_mode
        print(f"DEBUG: Initialized MultiLocalMaskingGenerator:")
        print(f"- Grid size: {self.T}x{self.H}x{self.W} ({self.total_patches} total patches)")
        print(f"- Patches per frame: {self.num_patches_per_frame}")
        print(f"- Masking mode: {masking_mode}")
        print(f"- Global mask ratio: {global_mask_ratio}")
        print(f"- Local mask ratio: {local_mask_ratio}")
        print(f"- Visible ratio: {visible_ratio}")
        print(f"- Number of local views: {num_local_views}")

    def generate_random_mask(self, mask_ratio):
        """Generate random mask"""
        # First create mask for one frame
        num_mask_per_frame = int(self.num_patches_per_frame * mask_ratio)
        perm = torch.randperm(self.num_patches_per_frame)
        mask_per_frame = torch.zeros(self.num_patches_per_frame, dtype=torch.bool)
        mask_per_frame[perm[:num_mask_per_frame]] = True
        
        # Repeat the same mask across all frames
        mask = mask_per_frame.repeat(self.T)  # Flatten and repeat
        
        return mask

    def generate_single_region_mask(self):
        """Generate region-based mask using pixel-space random crop logic"""
        # First create mask for one frame
        mask_per_frame = torch.ones(self.num_patches_per_frame, dtype=torch.bool)
        
        # Image dimensions in pixels
        img_H, img_W = self.H * 16, self.W * 16  # Convert patch grid to pixels
        
        # Calculate target visible area in pixels
        target_pixel_area = int(img_H * img_W * self.visible_ratio)
        region_side = int(torch.sqrt(torch.tensor(target_pixel_area)).item())
        
        # Calculate region size in pixels with some randomness
        h_size = min(img_H, max(32, region_side + torch.randint(-16, 17, (1,)).item()))
        w_size = min(img_W, max(32, region_side + torch.randint(-16, 17, (1,)).item()))
        
        # Random starting point in pixels
        h_start = torch.randint(0, max(1, img_H - h_size + 1), (1,)).item()
        w_start = torch.randint(0, max(1, img_W - w_size + 1), (1,)).item()
        
        # Calculate which patches overlap with this region
        patch_h_start = h_start // 16
        patch_w_start = w_start // 16
        patch_h_end = (h_start + h_size + 15) // 16  # +15 to round up
        patch_w_end = (w_start + w_size + 15) // 16

        # Create copy for loop version to compare
        # mask_per_frame_loop = mask_per_frame.clone()
        
        # Version 1: Original nested loops
        # for h in range(patch_h_start, min(patch_h_end, self.H)):
        #     for w in range(patch_w_start, min(patch_w_end, self.W)):
        #         patch_idx = h * self.W + w
        #         mask_per_frame_loop[patch_idx] = False
                
        # Version 2: Vectorized version
        h_indices = torch.arange(patch_h_start, min(patch_h_end, self.H))
        w_indices = torch.arange(patch_w_start, min(patch_w_end, self.W))
        h_grid, w_grid = torch.meshgrid(h_indices, w_indices, indexing='ij')
        patch_indices = h_grid * self.W + w_grid
        mask_per_frame[patch_indices.flatten()] = False
        
        # Verify both versions give same result
        # assert torch.all(mask_per_frame == mask_per_frame_loop), "Vectorized and loop versions differ!"
        # print("they are the same")
        # Tile the same mask across all frames
        mask = mask_per_frame.repeat(self.T)
        
        return mask

    def __call__(self):
        # print("\nDEBUG: Generating masks...")
        if self.masking_mode == 'random':
            global_mask = self.generate_random_mask(self.global_mask_ratio)
            local_masks = [self.generate_random_mask(self.local_mask_ratio) 
                          for _ in range(self.num_local_views)]
            
        elif self.masking_mode == 'region_based':
            global_mask = self.generate_single_region_mask()
            local_masks = [self.generate_single_region_mask() 
                          for _ in range(self.num_local_views)]
            
        else:  # mixed mode
            global_mask = self.generate_single_region_mask()
            local_masks = []
            for i in range(self.num_local_views):
                if i % 2 == 0:
                    local_masks.append(self.generate_random_mask(self.local_mask_ratio))
                else:
                    local_masks.append(self.generate_single_region_mask())

        # print("DEBUG: Final masks generated:")
        # print(f"- Global mask: {global_mask.sum()}/{self.total_patches} patches masked")
        # print(f"- Local masks: {len(local_masks)} views generated")
        # for i, mask in enumerate(local_masks):
        #     print(f"  View {i+1}: {mask.sum()}/{self.total_patches} patches masked")

        return global_mask, local_masks

class SparseTubeMaskingGenerator:
    def __init__(self, tokenizer_info, mask_ratio):
        """
        Args:
            tokenizer_info: Object with information about the tokenizer
            mask_ratio: Percentage of tokens to mask
        """
        self.mask_ratio = mask_ratio
        
        # Get token counts from tokenizer info
        if hasattr(tokenizer_info, 'tube_token_counts'):
            self.tube_token_counts = tokenizer_info.tube_token_counts
        else:
            # Fall back to calculating from video dimensions
            strides = tokenizer_info.strides
            offsets = tokenizer_info.offsets
            _, t, h, w = tokenizer_info.video_shape
            
            self.tube_token_counts = []
            for stride, offset in zip(strides, offsets):
                t_tokens = max(1, (t - offset[0]) // stride[0])
                h_tokens = max(1, (h - offset[1]) // stride[1])
                w_tokens = max(1, (w - offset[2]) // stride[2])
                token_count = t_tokens * h_tokens * w_tokens
                self.tube_token_counts.append(token_count)
        
        self.total_tokens = sum(self.tube_token_counts)
        # print(f"SparseTubeMaskingGenerator: total tokens={self.total_tokens}")
        self.num_masks = int(self.mask_ratio * self.total_tokens)
        # print(f"SparseTubeMaskingGenerator: num masks={self.num_masks}")
        # print(f"SparseTubeMaskingGenerator: total tokens={self.total_tokens}, tokens to mask={self.num_masks}")
        
    def __call__(self):
        # Create a properly sized mask
        mask = torch.zeros(self.total_tokens, dtype=torch.bool)
        
        # Randomly select tokens to mask
        indices = torch.randperm(self.total_tokens)[:self.num_masks]
        mask[indices] = True
        
        # print(f"DEBUG: Generated mask shape: {mask.shape}")
        # mask = mask.unsqueeze(0)
        
        return mask, []  # Return as tuple for consistency

    def __repr__(self):
        return f"SparseTubeMaskingGenerator(total={self.total_tokens}, mask={self.num_masks}, ratio={self.mask_ratio})" 

class MultiLocalSparseTubeMaskingGenerator:
    def __init__(self, tokenizer_info, global_mask_ratio, local_mask_ratio, num_local_views):
        """
        Args:
            tokenizer_info: Object with information about the tokenizer
            global_mask_ratio: Percentage of tokens to mask in global view
            local_mask_ratio: Percentage of tokens to mask in each local view
            num_local_views: Number of local views to generate
        """
        self.global_mask_ratio = global_mask_ratio
        self.local_mask_ratio = local_mask_ratio
        self.num_local_views = num_local_views
        
        # Get token counts from tokenizer info
        if hasattr(tokenizer_info, 'tube_token_counts'):
            self.tube_token_counts = tokenizer_info.tube_token_counts
        else:
            # Fall back to calculating from video dimensions
            strides = tokenizer_info.strides
            offsets = tokenizer_info.offsets
            _, t, h, w = tokenizer_info.video_shape
            
            self.tube_token_counts = []
            for stride, offset in zip(strides, offsets):
                t_tokens = max(1, (t - offset[0]) // stride[0])
                h_tokens = max(1, (h - offset[1]) // stride[1])
                w_tokens = max(1, (w - offset[2]) // stride[2])
                token_count = t_tokens * h_tokens * w_tokens
                self.tube_token_counts.append(token_count)
        
        self.total_tokens = sum(self.tube_token_counts)
        self.global_num_masks = int(self.global_mask_ratio * self.total_tokens)
        self.local_num_masks = int(self.local_mask_ratio * self.total_tokens)
        
    def __call__(self):
        # Generate global mask
        global_mask = torch.zeros(self.total_tokens, dtype=torch.bool)
        global_indices = torch.randperm(self.total_tokens)[:self.global_num_masks]
        global_mask[global_indices] = True
        
        # Generate local masks
        local_masks = []
        for _ in range(self.num_local_views):
            local_mask = torch.zeros(self.total_tokens, dtype=torch.bool)
            local_indices = torch.randperm(self.total_tokens)[:self.local_num_masks]
            local_mask[local_indices] = True
            local_masks.append(local_mask)
        
        return global_mask, local_masks

    def __repr__(self):
        return (f"MultiLocalSparseTubeMaskingGenerator(total={self.total_tokens}, "
                f"global_masks={self.global_num_masks}, local_masks={self.local_num_masks}, "
                f"num_views={self.num_local_views})") 