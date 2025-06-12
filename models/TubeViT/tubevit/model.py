from functools import partial
from typing import Any, Callable, List, Union

import lightning.pytorch as pl
import numpy as np
import torch
from torch import Tensor, nn, optim
from torch.nn import functional as F
from torchmetrics.functional import accuracy, f1_score
from torchvision.models.vision_transformer import EncoderBlock
from typing_extensions import OrderedDict
# import sys
# sys.path.append('/exafs1/well/noble/users/xpx456/codes/sigma_with_image/Sigma_with_image_dino/TubeViT')
from .positional_encoding import get_3d_sincos_pos_embed



class Encoder(nn.Module):
    """
    Transformer Model Encoder for sequence to sequence translation.
    Code from torch.
    Move pos_embedding to TubeViT
    """

    def __init__(
        self,
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float,
        attention_dropout: float,
        norm_layer: Callable[..., nn.Module] = partial(nn.LayerNorm, eps=1e-6),
    ):
        super().__init__()
        self.dropout = nn.Dropout(dropout)
        layers: OrderedDict[str, nn.Module] = OrderedDict()
        for i in range(num_layers):
            layers[f"encoder_layer_{i}"] = EncoderBlock(
                num_heads,
                hidden_dim,
                mlp_dim,
                dropout,
                attention_dropout,
                norm_layer,
            )
        self.layers = nn.Sequential(layers)
        self.ln = norm_layer(hidden_dim)

    def forward(self, x: Tensor):
        torch._assert(x.dim() == 3, f"Expected (batch_size, seq_length, hidden_dim) got {x.shape}")
        return self.ln(self.layers(self.dropout(x)))


class SparseTubesTokenizer(nn.Module):
    def __init__(self, hidden_dim, kernel_sizes, strides, offsets):
        super().__init__()
        self.hidden_dim = hidden_dim
        self.kernel_sizes = kernel_sizes
        self.strides = strides
        self.offsets = offsets
        self.tokenizer_type = 'sparse_tube'  # Add type marker
        
        # Add fallback values for code that might access these
        # but they're not actually used in sparse tube processing
        self._min_spatial_kernel = min([min(k[1], k[2]) for k in kernel_sizes])
        self._min_temporal_kernel = min([k[0] for k in kernel_sizes])
        
        self.conv_proj_weight = nn.Parameter(
            torch.empty((self.hidden_dim, 3, *self.kernel_sizes[0])).normal_(), requires_grad=True
        )

        self.register_parameter("conv_proj_weight", self.conv_proj_weight)

        self.conv_proj_bias = nn.Parameter(torch.zeros(len(self.kernel_sizes), self.hidden_dim), requires_grad=True)
        self.register_parameter("conv_proj_bias", self.conv_proj_bias)

    def forward(self, x: Tensor) -> Tensor:
        n, c, t, h, w = x.shape  # CTHW
        tubes = []
        for i in range(len(self.kernel_sizes)):
            if i == 0:
                weight = self.conv_proj_weight
            else:
                # Interpolate weights to match the current kernel size
                weight = F.interpolate(
                    self.conv_proj_weight, self.kernel_sizes[i], mode="trilinear", align_corners=False
                )

            # Adjust the input tensor based on the current offset
            tube_input = x[
                :, :, self.offsets[i][0] :, self.offsets[i][1] :, self.offsets[i][2] :
            ]

            # Perform 3D convolution
            tube = F.conv3d(
                tube_input,
                weight,
                bias=self.conv_proj_bias[i],
                stride=self.strides[i],
            )

            # Reshape to (N, hidden_dim, -1)
            tube = tube.reshape((n, self.hidden_dim, -1))

            tubes.append(tube)

        # Concatenate along the token dimension
        x = torch.cat(tubes, dim=-1)
        # Permute to (N, num_tokens, hidden_dim)
        x = x.permute(0, 2, 1).contiguous()
        return x

    # Add properties with warnings for compatibility
    @property
    def patch_size(self):
        print("WARNING: Accessing patch_size on SparseTubesTokenizer, which uses variable-sized kernels")
        return self._min_spatial_kernel
        
    @property
    def tubelet_size(self):
        print("WARNING: Accessing tubelet_size on SparseTubesTokenizer, which uses variable-sized kernels")
        return self._min_temporal_kernel


class SelfAttentionPooling(nn.Module):
    """
    Implementation of SelfAttentionPooling
    Original Paper: Self-Attention Encoding and Pooling for Speaker Recognition
    https://arxiv.org/pdf/2008.01077v1.pdf

    Code adapted from https://gist.github.com/pohanchi/c77f6dbfbcbc21c5215acde4f62e4362
    """

    def __init__(self, input_dim):
        super(SelfAttentionPooling, self).__init__()
        self.W = nn.Linear(input_dim, 1)

    def forward(self, x):
        """
        input:
            x : size (N, T, H), N: batch size, T: sequence length, H: Hidden dimension

        attention_weight:
            att_w : size (N, T, 1)

        return:
            utter_rep: size (N, H)
        """

        # (N, T, H) -> (N, T) -> (N, T, 1)
        att_w = F.softmax(self.W(x).squeeze(dim=-1), dim=-1).unsqueeze(dim=-1)
        x = torch.sum(x * att_w, dim=1)
        return x


class TubeViT(nn.Module):
    def __init__(
        self,
        num_classes: int,
        video_shape: Union[List[int], np.ndarray],  # CTHW
        num_layers: int,
        num_heads: int,
        hidden_dim: int,
        mlp_dim: int,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        representation_size=None,
        no_attention_pooling: bool = False,
        no_head: bool = False,
    ):
        super(TubeViT, self).__init__()
        self.video_shape = np.array(video_shape)  # CTHW
        self.num_classes = num_classes
        self.hidden_dim = hidden_dim
        self.kernel_sizes = (
            (8, 8, 8),
            (16, 4, 4),
            (4, 12, 12),
            (1, 16, 16),
        )

        self.strides = (
            (16, 32, 32),
            (6, 32, 32),
            (16, 32, 32),
            (32, 16, 16),
        )

        self.offsets = (
            (0, 0, 0),
            (4, 8, 8),
            (0, 16, 16),
            (0, 0, 0),
        )
        self.sparse_tubes_tokenizer = SparseTubesTokenizer(
            self.hidden_dim, self.kernel_sizes, self.strides, self.offsets
        )

        # Initialize pos_embedding as a buffer; it will be generated dynamically
        # self.register_buffer("pos_embedding", None)
       
        self.class_token = nn.Parameter(torch.zeros(1, 1, self.hidden_dim), requires_grad=True)
        self.register_parameter("class_token", self.class_token)

        self.pos_embeddings = nn.ParameterDict()
        # import pdb
        # pdb.set_trace();
        c,t,h,w = video_shape
        video_shape_all = [(c,t,h,w),(c,t,96,96)]
        # print('VIDEO_SHAPE_ALL',video_shape_all)
        for idx, shape in enumerate(video_shape_all):
            shape_tuple = tuple(shape[1:])  # (T, H, W)
            self.video_shape = np.array(shape)
            pos_embed = self._generate_position_embedding()
            self.pos_embeddings[str(shape_tuple)] = pos_embed 
        # print(self.pos_embeddings)
        # Add a class token

        self.encoder = Encoder(
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=self.hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
        )

        # Initialize attention pooling only if no_attention_pooling is False
        if not no_attention_pooling:
            self.attention_pooling = SelfAttentionPooling(self.hidden_dim)
        else:
            self.attention_pooling = None

        heads_layers: OrderedDict[str, nn.Module] = OrderedDict()
        if not no_head:
            if representation_size is None:
                heads_layers["head"] = nn.Linear(self.hidden_dim, self.num_classes)
            else:
                heads_layers["pre_logits"] = nn.Linear(self.hidden_dim, representation_size)
                heads_layers["act"] = nn.Tanh()
                heads_layers["head"] = nn.Linear(representation_size, self.num_classes)

            self.heads = nn.Sequential(heads_layers)
        else:
            self.heads = None

        # Initialize positional embeddings based on the initial video shape
        # self.update_positional_embedding(self.video_shape)

    def forward(self, x, drop_ratio=None):
        current_video_shape = x.shape[2:]
        
        x = self.sparse_tubes_tokenizer(x)
        n = x.shape[0]  # batch size
        
        batch_class_token = self.class_token.expand(n, -1, -1)
        x = torch.cat([batch_class_token, x], dim=1)
        
        x = x + self.pos_embeddings[str(tuple((current_video_shape)))]
        
        if drop_ratio is not None:
            # Calculate number of tokens to keep (excluding class token)
            num_tokens = x.shape[1] - 1  # -1 for class token
            num_tokens_to_keep = int(num_tokens * (1 - drop_ratio))
            
            # Create indices for all tokens after CLS
            token_range = torch.arange(1, x.shape[1], device=x.device)
            
            # Efficient batch random selection
            # Shape: [batch_size, num_tokens_to_keep]
            rand_indices = torch.stack([
                token_range[torch.randperm(num_tokens, device=x.device)[:num_tokens_to_keep]]
                for _ in range(n)
            ])
            
            # Add CLS token index (0) at the start
            # Shape: [batch_size, num_tokens_to_keep + 1]
            keep_indices = torch.cat([
                torch.zeros(n, 1, dtype=torch.long, device=x.device),
                rand_indices
            ], dim=1)
            
            # Create batch indices
            # Shape: [batch_size, num_tokens_to_keep + 1]
            batch_indices = torch.arange(n, device=x.device).unsqueeze(1).expand(-1, num_tokens_to_keep + 1)
            
            # Gather selected tokens in one operation
            x = x[batch_indices, keep_indices]
        
        x = self.encoder(x)
        
        if self.attention_pooling is not None:
            x = self.attention_pooling(x)
        
        if self.heads is not None:
            x = self.heads(x)
        
        return x

    def _calc_conv_shape(self, kernel_size, stride, offset) -> np.ndarray:
        kernel_size = np.array(kernel_size)
        stride = np.array(stride)
        offset = np.array(offset)
        # Calculate output shape: floor((input - offset - kernel_size) / stride + 1)
        # print('SELF_VIDEO_SHAPE',self.video_shape)
        output = np.floor(((self.video_shape[[1, 2, 3]] - offset - kernel_size) / stride) + 1).astype(int)
        return output

    def _generate_position_embedding(self) -> torch.Tensor:
        """
        Generate positional embeddings based on the current video shape.

        Returns:
            torch.Tensor: Positional embeddings of shape (1, num_tokens, hidden_dim).
        """
        device = self.class_token.device
        position_embedding = [torch.zeros(1, 1, self.hidden_dim, device=device)]  # Class token embedding

        for i in range(len(self.kernel_sizes)):
            tube_shape = self._calc_conv_shape(self.kernel_sizes[i], self.strides[i], self.offsets[i])
            pos_embed = get_3d_sincos_pos_embed(
                embed_dim=self.hidden_dim,
                tube_shape=tube_shape,
                kernel_size=self.kernel_sizes[i],
                stride=self.strides[i],
                offset=self.offsets[i],
            )
            pos_embed = pos_embed.to(device)
            pos_embed = pos_embed.unsqueeze(0)  # Add batch dimension
            position_embedding.append(pos_embed)

        position_embedding = torch.cat(position_embedding, dim=1).contiguous()  # (1, num_tokens, hidden_dim)
        return position_embedding

    def update_positional_embedding(self, new_video_shape):
        """
        Update the positional embeddings based on the new video shape.

        Args:
            new_video_shape (Union[List[int], np.ndarray]): The new video shape (CTHW).
        """
        self.video_shape = np.array(new_video_shape)
        new_pos_embed = self._generate_position_embedding()
        self.pos_embedding = new_pos_embed  # Update the buffer


class TubeViTLightningModule(pl.LightningModule):
    def __init__(
        self,
        num_classes,
        video_shape,
        num_layers,
        num_heads,
        hidden_dim,
        mlp_dim,
        lr: float = 3e-4,
        weight_decay: float = 0,
        weight_path: str = None,
        max_epochs: int = None,
        label_smoothing: float = 0.0,
        dropout: float = 0.0,
        attention_dropout: float = 0.0,
        no_attention_pooling: bool = False,
        no_head: bool = False,
        **kwargs,
    ):
        super().__init__()
        self.save_hyperparameters()
        self.num_classes = num_classes
        self.model = TubeViT(
            num_classes=num_classes,
            video_shape=video_shape,
            num_layers=num_layers,
            num_heads=num_heads,
            hidden_dim=hidden_dim,
            mlp_dim=mlp_dim,
            dropout=dropout,
            attention_dropout=attention_dropout,
            no_attention_pooling=no_attention_pooling,
            no_head=no_head,
        )

        self.lr = lr
        self.loss_func = nn.CrossEntropyLoss(label_smoothing=label_smoothing)
        self.example_input_array = torch.zeros(1, *video_shape)

        if weight_path is not None:
            self.model.load_state_dict(torch.load(weight_path), strict=False)
        self.max_epochs = max_epochs
        self.weight_decay = weight_decay

    def forward(self, x):
        return self.model(x)

    def training_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = torch.softmax(y_hat, dim=-1)

        # Logging to TensorBoard by default
        self.log("train_loss", loss, prog_bar=True, on_step=True, on_epoch=True)
        self.log(
            "train_acc",
            accuracy(y_pred, y, task="multiclass", num_classes=self.num_classes),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )
        self.log(
            "train_f1",
            f1_score(y_pred, y, task="multiclass", num_classes=self.num_classes),
            prog_bar=True,
            on_step=True,
            on_epoch=True,
        )

        return loss

    def validation_step(self, batch, batch_idx):
        x, y = batch
        y_hat = self(x)

        loss = self.loss_func(y_hat, y)

        y_pred = torch.softmax(y_hat, dim=-1)

        # Logging to TensorBoard by default
        self.log("val_loss", loss, prog_bar=True, on_step=False, on_epoch=True)
        self.log(
            "val_acc",
            accuracy(y_pred, y, task="multiclass", num_classes=self.num_classes),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )
        self.log(
            "val_f1",
            f1_score(y_pred, y, task="multiclass", num_classes=self.num_classes),
            prog_bar=True,
            on_step=False,
            on_epoch=True,
        )

        return loss

    def on_train_epoch_end(self) -> None:
        if self.trainer is not None and self.trainer.optimizers is not None:
            optimizer = self.trainer.optimizers[0]
            current_lr = optimizer.param_groups[0]["lr"]
            self.log("lr", current_lr, on_step=False, on_epoch=True)

    def configure_optimizers(self):
        optimizer = optim.Adam(self.model.parameters(), lr=self.lr, weight_decay=self.weight_decay)
        if self.max_epochs is not None:
            # Assuming total_steps is max_epochs * steps_per_epoch
            # You might need to adjust this based on your DataLoader
            total_steps = self.max_epochs
            lr_scheduler = optim.lr_scheduler.OneCycleLR(
                optimizer=optimizer, max_lr=self.lr, total_steps=total_steps
            )
            return [optimizer], [lr_scheduler]
        else:
            return optimizer

    def predict_step(self, batch: Any, batch_idx: int, dataloader_idx: int = 0) -> Any:
        x, y = batch
        y_hat = self(x)
        y_pred = torch.softmax(y_hat, dim=-1)

        return {"y": y, "y_pred": torch.argmax(y_pred, dim=-1), "y_prob": y_pred}


# Example Usage
if __name__ == "__main__":
    # Define model parameters
    num_classes = 10
    video_shape = [3, 64, 128, 128]  # C, T, H, W
    num_layers = 6
    num_heads = 8
    hidden_dim = 512
    mlp_dim = 1024
    lr = 3e-4
    weight_decay = 1e-4
    max_epochs = 10
    label_smoothing = 0.1
    dropout = 0.1
    attention_dropout = 0.1
    no_attention_pooling = False  # Set to True to disable attention pooling
    no_head = False  # Set to True to disable the classification head
    representation_size = None  # Or specify an integer

    # Instantiate the Lightning Module
    model = TubeViTLightningModule(
        num_classes=num_classes,
        video_shape=video_shape,
        num_layers=num_layers,
        num_heads=num_heads,
        hidden_dim=hidden_dim,
        mlp_dim=mlp_dim,
        lr=lr,
        weight_decay=weight_decay,
        max_epochs=max_epochs,
        label_smoothing=label_smoothing,
        dropout=dropout,
        attention_dropout=attention_dropout,
        no_attention_pooling=no_attention_pooling,
        no_head=no_head,
    )

    # Create dummy data for demonstration
    batch_size = 8
    dummy_x = torch.randn(batch_size, *video_shape)  # (N, C, T, H, W)
    dummy_y = torch.randint(0, num_classes, (batch_size,))  # (N,)

    # Create a DataLoader
    dataset = torch.utils.data.TensorDataset(dummy_x, dummy_y)
    dataloader = torch.utils.data.DataLoader(dataset, batch_size=batch_size)

    # Instantiate a Trainer
    trainer = pl.Trainer(
        max_epochs=2,
        gpus=0,  # Set to 1 if using GPU
        progress_bar_refresh_rate=20,
    )

    # Train the model
    trainer.fit(model, dataloader, dataloader)

    # Make predictions
    predictions = trainer.predict(model, dataloader)
    print(predictions)
