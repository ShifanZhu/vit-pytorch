import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, hidden_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(hidden_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        return self.net(x)

class Attention(nn.Module):
    def __init__(self, dim, heads = 8, dim_head = 64, dropout = 0.):
        super().__init__()
        inner_dim = dim_head *  heads
        project_out = not (heads == 1 and dim_head == dim)

        self.heads = heads
        self.scale = dim_head ** -0.5

        self.norm = nn.LayerNorm(dim)

        self.attend = nn.Softmax(dim = -1)
        self.dropout = nn.Dropout(dropout)

        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        x = self.norm(x)

        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)
        attn = self.dropout(attn)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

# Transformer encoder
# Parameters:
#   dim: embedding dimension, e.g. 512
#   depth: number of layers, e.g. 6
#   heads: number of attention heads, e.g. 8
#   dim_head: dimension per attention head, e.g. 64
#   mlp_dim: hidden dimension of MLP, e.g. 2048
#   dropout: dropout probability, e.g. 0.1

# 📐 Tensor Shapes Example
# Suppose:
#   batch_size = 2
#   seq_len = 197 (CLS + 196 patches in ViT)
#   dim = 768
# Flow:
#   Input x: (2, 197, 768)
#   Attention: (2, 197, 768)
#   Residual: (2, 197, 768)
#   FeedForward: (2, 197, 768)
#   Residual: (2, 197, 768)
# After all layers → (2, 197, 768)
class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout),
                FeedForward(dim, mlp_dim, dropout = dropout)
            ]))

    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x

        return self.norm(x)

# Vision Transformer (ViT)
# 1. Patch embedding:
#   Image → patches → flatten → linear projection → embedding.
# 2. CLS token:
#   A special learnable token prepended to sequence; final representation used for classification.
# 3. Positional encoding:
#   Adds position info since Transformer has no inherent sense of order.
# 4. Transformer encoder:
#   Processes sequence of patch embeddings (like tokens in NLP).
# 5. Pooling:
#   "cls" → take CLS token.
#   "mean" → average over all patch embeddings.
# 6. Head:
#   Linear layer projects to number of classes.
class ViT(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim,
                 pool='cls', channels=3, dim_head=64, dropout=0., emb_dropout=0.):
        """
        Args:
            image_size: input image size (int or tuple, e.g., 224)
            patch_size: size of each patch (int or tuple, e.g., 16)
            num_classes: number of output classes
            dim: embedding dimension for transformer
            depth: number of transformer layers
            heads: number of attention heads
            mlp_dim: hidden dimension of MLP inside transformer
            pool: pooling method ('cls' token or 'mean')
            channels: number of image channels (3 for RGB)
            dim_head: dimension per attention head
            dropout: dropout probability inside transformer
            emb_dropout: dropout probability after embeddings
        """
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        # Ensure image can be evenly divided into patches
        assert image_height % patch_height == 0 and image_width % patch_width == 0, \
            'Image dimensions must be divisible by the patch size.'

        # Number of patches = (H / patch_size) * (W / patch_size)
        # e.g. 224//16  ->  14 * 14 = 196
        num_patches = (image_height // patch_height) * (image_width // patch_width)

        # Each patch flattened dimension = channels * patch_height * patch_width
        # e.g. 3 * 16 * 16 = 768
        patch_dim = channels * patch_height * patch_width

        # Pooling must be either CLS token or mean pooling
        #   If using CLS token, it will be the first token in the sequence
        #   If using mean pooling, it will average all (197) tokens
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        # --- Patch Embedding ---
        # Rearrange image into patches -> Linear projection -> Normalize
        self.to_patch_embedding = nn.Sequential(
            # b = batch, c = channels(3), h = height(16), w = width(16)
            # Splits image into patches, flattens patches
            #          (h w) (p1 p2 c) can be 14*14 * 16*16*3 where 14*14 is patch num and 16*16*3 is patch dim
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_height, p2=patch_width),
            nn.LayerNorm(patch_dim), # 16*16*3 = 768
            nn.Linear(patch_dim, dim),  # project to transformer dimension. e.g. 768 -> 512
            nn.LayerNorm(dim),
        )

        # --- Positional Encoding & CLS Token ---
        # In ViT, pos_embedding is initialized randomly because it is learned during training.
        # This works well since:
        #   Patch positions are fixed in number.
        #   The model can optimize positional encodings for image data.
        # In NLP Transformers, sinusoidal encodings were used originally for generalization to arbitrary sequence lengths.
        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))  # (1, N+1, D)  e.g. 1 * 197 * 512

        # cls_token = a special learnable vector prepended to patch embeddings.
        # Role: learns to gather global information from all patches, and its final embedding is used for classification.
        # Analogy: like the [CLS] token in BERT, but for images.
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))  # classification token  e.g. 1 * 1 * 512
        self.dropout = nn.Dropout(emb_dropout)

        # --- Transformer Encoder ---
        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        # --- Classification Head ---
        self.pool = pool  # 'cls' or 'mean'
        self.to_latent = nn.Identity()
        self.mlp_head = nn.Linear(dim, num_classes) # e.g. 512 -> 1000

    def forward(self, img):
        """
        Args:
            img: input tensor (batch_size, channels, height, width)

        Returns:
            logits: (batch_size, num_classes)
        """
        # 1. Convert image to patch embeddings
        x = self.to_patch_embedding(img)   # (b, num_patches, dim)  e.g. b * 196 * 512
        b, n, _ = x.shape

        # 2. Add CLS token (prepended for classification)
        cls_tokens = repeat(self.cls_token, '1 1 d -> b 1 d', b=b)  # (b, 1, dim)  e.g. 1 * 1 * 512 -> b * 1 * 512
        x = torch.cat((cls_tokens, x), dim=1)  # (b, n+1, dim)  e.g. b * 1 * 512 + b * 196 * 512 -> b * 197 * 512

        # 3. Add positional encoding
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        # 4. Pass through Transformer encoder
        x = self.transformer(x)  # (b, n+1, dim)  e.g. b * 197 * 512

        # After passing through the Transformer, we have: x.shape = (batch_size, seq_len, dim) where:
        #   seq_len = num_patches + 1 (patch embeddings + CLS token)
        #   dim = embedding dimension (e.g., 768 in ViT-Base, 512 here).
        # Now we must collapse this sequence into a single vector per image, since classification needs one vector per sample.
        
        # Two options
        # 1. CLS token pooling (self.pool == 'cls')
        #   Take the first token (x[:, 0]), which is the CLS token.
        #   Shape: (batch_size, dim)
        #   This works because the CLS token was designed to aggregate information from all patches during self-attention.
        # 2. Mean pooling (self.pool == 'mean')
        #   Average across all tokens: x.mean(dim=1).
        #   Shape: (batch_size, dim)
        #   This treats the average of all patch embeddings as the global representation.

        # 5. Pooling: take CLS token or mean
        x = x.mean(dim=1) if self.pool == 'mean' else x[:, 0]

        # 6. Final classification head 
        # Latent projection
        # Currently, self.to_latent is nn.Identity(). This means currently nothing happens here (it just passes through).
        # But this placeholder is often used if we want to insert another projection later 
        # (e.g., normalize, project to a latent space for self-supervised tasks like DINO/SimCLR).
        x = self.to_latent(x)

        # Classification head
        # Maps the pooled embedding → logits over classes.
        # Output shape: (batch_size, num_classes)
        # This is the final prediction (e.g., 1000 classes for ImageNet).
        return self.mlp_head(x)  # (b, num_classes)