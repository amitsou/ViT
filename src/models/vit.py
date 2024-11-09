"""Implementation of the Vision Transformer model."""

import os
import sys
import yaml
import einops
import tqdm
import torch
import torchvision
import torch.optim as optim
from torch import nn
from torchsummary import summary
from torchvision.transforms import (
    Compose, Resize, ToTensor, Normalize,
    RandomHorizontalFlip, RandomHorizontalFlip, RandomCrop
)

from src.utils.model_utils import ModelUtils

config_path = os.path.join(
    os.path.dirname(__file__),
    '..',
    'config',
    'vit_config.yaml'
)
config = ModelUtils().load_model_config(config_path)

#Model parameters
PATCH_SIZE = config['model']['PATCH_SIZE']
LATENT_SIZE = config['model']['LATENT_SIZE']
N_CHANNELS = config['model']['N_CHANNELS']
N_HEADS = config['model']['N_HEADS']
N_ENCODER_LAYERS = config['model']['N_ENCODER_LAYERS']
DROPOUT = config['model']['DROPOUT']
N_CLASSES = config['model']['N_CLASSES']
SIZE = config['model']['SIZE']

#Training parameters
EPOCHS = config['training']['EPOCHS']
BASE_LR = config['training']['BASE_LR']
WEIGHT_DECAY = config['training']['WEIGHT_DECAY']
BATCH_SIZE = config['training']['BATCH_SIZE']

device = ModelUtils().get_device()

#Implementation of the Linear Projection layer
class InputEmbedding(nn.Module):
    def __init__(
        self,
        patch_size=PATCH_SIZE,
        n_channels=N_CHANNELS,
        device=device,
        latent_size=LATENT_SIZE,
        batch_size=BATCH_SIZE
    ) -> None:
        super(InputEmbedding, self).__init__()

        self.patch_size = patch_size
        self.n_channels = n_channels
        self.device = device
        self.latent_size = latent_size
        self.batch_size = batch_size
        self.input_size = patch_size * patch_size * n_channels # 3 * 16 * 16 = 768 (for CIFAR-10)

        #Linear Projection 1 x 768 x 768
        self.linear_projection = nn.Linear(self.input_size, self.latent_size)
        #Class Token 1 x 768
        self.class_token = nn.Parameter(torch.randn(self.batch_size, 1, self.latent_size)).to(self.device)
        #Positional Embedding 1 x 257 x 768
        self.pos_embedding = nn.Parameter(torch.randn(1, (self.patch_size ** 2) + 1, self.latent_size)).to(self.device)

    def forward(self, input_data) -> torch.Tensor:
        """
        Forward pass for the Vision Transformer (ViT) model.
        Args:
            input_data (torch.Tensor): Input tensor of shape (batch_size, channels, height, width).
        Returns:
            torch.Tensor: Output tensor after linear projection and positional embedding addition.
        """
        input_data = input_data.to(self.device)

        #Patchify the input image
        patches = einops.rearrange(
            input_data,
            'b c (h h1) (w w1) -> b (h w) (h1 w1 c)',
            h1=self.patch_size,
            w1=self.patch_size
        )

        #Linear Projection
        linear_projection = self.linear_projection(patches).to(self.device)
        b, n, _ = linear_projection.shape # b = batch size, n = number of patches, _ = latent size
        linear_projection = torch.cat((self.class_token, linear_projection), dim=1)
        pos_embedding = einops.repeat(self.pos_embedding, 'b 1 d -> b n d', m=n+1)
        linear_projection += pos_embedding

        return linear_projection

class EncoderBlock(nn.module):
    def __init__(
        self,
        latent_size=LATENT_SIZE,
        n_heads=N_HEADS,
        device=device,
        dropout=DROPOUT
        ) -> None:
        super(EncoderBlock, self).__init__()

        self.latent_size = latent_size
        self.n_heads = n_heads
        self.device = device
        self.dropout = dropout

        #Normalization layer
        self.norm = nn.LayerNorm(self.latent_size)

        #Multi-head attention layer
        self.multihead = nn.MultiheadAttention(
            embed_dim=self.latent_size,
            num_heads=self.n_heads,
            dropout=self.dropout
        )

        #MLP layer
        self.mlp = nn.Sequential(
            nn.Linear(self.latent_size, 4 * self.latent_size),
            nn.GELU(),
            nn.DropOut(self.dropout),
            nn.Linear(4 * self.latent_size, self.latent_size),
            nn.DropOut(self.dropout)
        )

    def forward(self, embedded_patches):
        first_norm_out = self.norm1(embedded_patches)
        attention_out = self.multihead(first_norm_out, first_norm_out, first_norm_out)[0]

        #First residual connection
        first_added = attention_out + embedded_patches

        #Second normalization layer
        second_norm_out = self.norm(first_added)
        ff_out = self.MLP(second_norm_out) #mlp output ff=feed forward

        return ff_out + first_added

#Putting it all together
class ViT(nn.module):
    def __init__(
        self,
        num_encoders=N_ENCODER_LAYERS,
        latent_size=LATENT_SIZE,
        device=device,
        n_classes=N_CLASSES,
        dropout=DROPOUT
    ) -> None:
        super(ViT, self).__init__()

        self.num_encoders = num_encoders
        self.latent_size = latent_size
        self.device = device
        self.n_classes = n_classes
        self.dropout = dropout

        self.embedding = InputEmbedding()

        #Create the stack of encoders
        self.encoders_stack = nn.ModuleList([
            EncoderBlock() for _ in range(self.num_encoders)
        ])

        self.MLP_head = nn.Sequential(
            nn.LayerNorm(self.latent_size),
            nn.Linear(self.latent_size, self.latent_size),
            nn.Linear(self.latent_size, self.n_classes)
        )

    def forward(self, test_input):
        encoder_output = self.embedding(test_input)

        for encoder_layer in self.encoders_stack:
            encoder_output = encoder_layer(encoder_output)

        cls_token_embedding = encoder_output[:, 0] #Extract the class token

        return self.MLP_head(cls_token_embedding)
