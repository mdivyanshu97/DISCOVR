import os
from torchvision import transforms
from discovr.data.transforms import transforms as video_transforms
from discovr.data.transforms import volume_transforms
from discovr.models.masking_generator import TubeMaskingGenerator, MultiLocalMaskingGenerator, SparseTubeMaskingGenerator
from discovr.data.datasets.kinetics import VideoClsDataset, VideoMAE
from discovr.data.transforms.transforms import  *

import torch


class DataAugmentationForVideoMAE(object):
    def __init__(self, args):
        self.grayscale_mode = getattr(args, 'grayscale_mode', False)
        
        if self.grayscale_mode:
            self.input_mean = [0.5]
            self.input_std = [0.5]
        else:
            self.input_mean = [0.485, 0.456, 0.406]
            self.input_std = [0.229, 0.224, 0.225]
            
        normalize = GroupNormalize(mean=self.input_mean, std=self.input_std)
        
        if self.grayscale_mode:
            gray_scale = GroupGrayscale()
        else:
            gray_scale = GroupRandomGrayScale(p=args.gray_scale_prob)
            
        if args.augmentation == 'resize':
            self.train_augmentation = GroupResize(args.input_size)
        else:
            self.train_augmentation = GroupMultiScaleCrop(args.input_size, [1, .875, .75, .66])

        self.transform = transforms.Compose([                            
            gray_scale,
            self.train_augmentation,
            Stack(roll=False),
            ToTorchFormatTensor(div=True),
            normalize,
        ])
        
        self.tokenizer_type = getattr(args, 'tokenizer_type', 'default')
        
        if args.tokenizer_type == 'sparse_tube':
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
            
            num_frames = args.num_frames
            height = args.input_size
            width = args.input_size
            
            tube_token_counts = []
            total_tokens = 0
            
            for i, (kernel, stride, offset) in enumerate(zip(kernel_sizes, strides, offsets)):
                t_tokens = max(1, (num_frames - offset[0] - kernel[0] + 1 + stride[0] - 1) // stride[0])
                h_tokens = max(1, (height - offset[1] - kernel[1] + 1 + stride[1] - 1) // stride[1])
                w_tokens = max(1, (width - offset[2] - kernel[2] + 1 + stride[2] - 1) // stride[2])
                
                token_count = t_tokens * h_tokens * w_tokens
                tube_token_counts.append(token_count)
                total_tokens += token_count
            
            tokenizer_info = type('TokenizerInfo', (), {
                'strides': strides,
                'offsets': offsets,
                'video_shape': (3, num_frames, height, width),
                'tube_token_counts': tube_token_counts
            })
            
            self.masked_position_generator = SparseTubeMaskingGenerator(
                tokenizer_info,
                args.mask_ratio
            )
        elif args.mask_type == 'multi_local':
            self.masked_position_generator = MultiLocalMaskingGenerator(
                window_size=args.window_size,
                global_mask_ratio=args.global_mask_ratio,
                local_mask_ratio=args.local_mask_ratio,
                num_local_views=args.num_local_views,
                masking_mode=args.masking_mode,
                visible_ratio=args.visible_ratio
            )
        elif args.mask_type in ['tube', 'parts', 'tube_fgbg']:
            self.masked_position_generator = TubeMaskingGenerator(args.window_size, args.mask_ratio)
        else:
            raise ValueError(f"Unknown mask type: {args.mask_type}")

    def __call__(self, images):
        process_data, _ = self.transform(images)
        global_mask, local_masks = self.masked_position_generator()
        return process_data, (global_mask, local_masks)

    def __repr__(self):
        repr_str = "(DataAugmentationForVideoMAE,\n"
        repr_str += f"  transform = {self.transform},\n"
        repr_str += f"  Masked position generator = {self.masked_position_generator},\n"
        repr_str += ")"
        return repr_str


def build_pretraining_dataset(args):
    transform = DataAugmentationForVideoMAE(args)
    dataset = VideoMAE(
        root=args.data_path,
        test_mode=args.test_mode if args.test_mode is not None else False,
        setting=args.data_path_csv,
        target_type=args.target_type,
        mask_ratio=args.mask_ratio,
        video_ext='avi',
        is_color=True,
        modality='rgb',
        new_length=args.num_frames,
        new_step=args.sampling_rate,
        transform=transform,
        temporal_jitter=False,
        video_loader=True,
        use_decord=True,
        use_torchcodec=args.use_torchcodec,
        lazy_init=False,
        use_dino_crop=args.use_dino_crop,
        num_local_views=args.num_local_views,
        local_size=args.local_size,
        grayscale_mode=args.grayscale_mode)
    return dataset


def build_dataset(is_train, test_mode, args):
    if args.data_set == 'Kinetics-400':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = args.data_path_csv
            data_path = args.data_path
        elif test_mode is True:
            mode = 'test'
            anno_path = args.data_path_csv
            data_path = args.data_path
        else:  
            mode = 'validation'
            anno_path = args.data_path_csv
            data_path = args.data_path

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path=data_path,
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=args.test_num_crop,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=args.input_size,
            new_width=args.input_size,
            args=args)
        nb_classes = args.nb_classes
    
    elif 'SSV2' in args.data_set:
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            if 'mini' in args.data_set:
                anno_path = os.path.join(args.data_path, 'train_mini.csv')
            else:
                anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 
        dataset = SSVideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=1,
            num_segment=args.num_frames,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 174

    elif args.data_set == 'UCF101':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path)
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path)
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path)

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 101
    elif args.data_set == 'GYM99' or args.data_set == 'FXS1' or args.data_set == 'UBS1':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path)
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path)
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path)

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        if args.data_set =='GYM99':
            nb_classes = 99  
        elif args.data_set =='FXS1':
            nb_classes = 11  
        elif args.data_set =='UBS1':
            nb_classes = 15  

    elif args.data_set == 'DIV48':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path)
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path)
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path)

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 48
    
    elif args.data_set == 'HMDB51':
        mode = None
        anno_path = None
        if is_train is True:
            mode = 'train'
            anno_path = os.path.join(args.data_path, 'train.csv')
        elif test_mode is True:
            mode = 'test'
            anno_path = os.path.join(args.data_path, 'test.csv') 
        else:  
            mode = 'validation'
            anno_path = os.path.join(args.data_path, 'val.csv') 

        dataset = VideoClsDataset(
            anno_path=anno_path,
            data_path='/',
            mode=mode,
            clip_len=args.num_frames,
            frame_sample_rate=args.sampling_rate,
            num_segment=1,
            test_num_segment=args.test_num_segment,
            test_num_crop=args.test_num_crop,
            num_crop=1 if not test_mode else 3,
            keep_aspect_ratio=True,
            crop_size=args.input_size,
            short_side_size=args.short_side_size,
            new_height=256,
            new_width=320,
            args=args)
        nb_classes = 51
    else:
        raise NotImplementedError()
    print("Number of the class = %d" % args.nb_classes)

    return dataset, nb_classes
