import argparse
from pathlib import Path

import torch
import models.modeling_pretrain as modeling_pretrain 


def clean_state_dict(state_dict):
    """Strip distributed/backbone prefixes so keys match `create_model`."""
    cleaned = {}
    for key, value in state_dict.items():
        if key.startswith("module."):
            key = key[len("module.") :]
        elif key.startswith("backbone."):
            key = key[len("backbone.") :]
        cleaned[key] = value
    return cleaned


def build_model(args):
    return modeling_pretrain.pretrain_videomae_base_patch16_224(
        pretrained=False,
        img_size=112,
        num_frames=64,
        tokenizer_type="default",
        mask_type="multi_local",
        mask_ratio=0.9,
        qkv_bias=False,
        num_local_views=4,
        loss_func="SWAV",
        num_prototypes=3000,
        sinkhorn_iterations=10,
        eps=0.05,
        kwindow=1,
        use_combined_dino_swav=True,
        skip_dino_loss=False,
        use_video_dino=True,
        use_dino_crop=False,
        local_size=96,
        use_turbo_training=False,
        turbo_recon_ratio=0.25,
        dino_out_dim=16384,
        dino_hidden_dim=2048,
        dino_bottleneck_dim=256,
        decoder_depth=4,
        decoder_num_classes=768,
        use_mean_pooling=False,
        use_checkpoint=False,

    )


def main():
    parser = argparse.ArgumentParser(
        description="Load DISCOVR encoder and run a dummy tensor through it."
    )
    parser.add_argument(
        "--repo-root",
        type=Path,
        default=Path("."),
        help="Root directory that contains the DISCOVR repo (default: current directory).",
    )
    parser.add_argument(
        "--checkpoint",
        type=Path,
        default=None,
        help="Path to DISCOVR checkpoint. Overrides repo-root/checkpoint-path defaults.",
    )
    parser.add_argument("--num_frames", type=int, default=64, help="Number of temporal frames.")
    parser.add_argument("--batch_size", type=int, default=2, help="Dummy batch size.")
    parser.add_argument("--device", type=str, default="cuda", help="Device for inference.")
    parser.add_argument(
        "--checkpoint-file",
        type=Path,
        default=Path("path/to/checkpoint-799.pth"),
        help="Checkpoint path relative to --repo-root when --checkpoint is not set.",
    )
    args = parser.parse_args()

    print(f"Loading checkpoint from {args.checkpoint}")
    checkpoint_path = args.checkpoint or args.repo_root / args.checkpoint_file
    print(f"Resolved checkpoint path: {checkpoint_path}")
    checkpoint = torch.load(checkpoint_path, map_location="cpu", weights_only=False)
    state_dict = checkpoint.get("model", checkpoint)
    state_dict = clean_state_dict(state_dict)

    model = build_model(args)
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing:
        print("Missing keys (will be ignored):", missing)
    if unexpected:
        print("Unexpected keys (dropped):", unexpected)

    if hasattr(model, "module"):
        module_model = model.module
    else:
        module_model = model

    if hasattr(module_model, "video_teacher"):
        encoder = module_model.video_teacher
        print("Using video_teacher for inference")
    else:
        encoder = module_model.encoder if hasattr(module_model, "encoder") else module_model
        print("Using encoder for inference")

    encoder = encoder.to(args.device)
    encoder.eval()

    dummy = torch.randn(
        args.batch_size, 3, args.num_frames, 112, 112, dtype=torch.float32, device=args.device
    )
    num_patches = encoder.patch_embed.num_patches
    mask = torch.zeros(args.batch_size, num_patches, dtype=torch.bool, device=args.device)

    with torch.no_grad():
        output = encoder(dummy, mask)[0]
        cls = output[:, 0, :]
        print("CLS token shape:", cls.shape)


if __name__ == "__main__":
    main()

