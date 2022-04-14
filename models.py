import torch
from torch import nn

from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# helpers

def pair(t):
    return t if isinstance(t, tuple) else (t, t)

# classes

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn
    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class FeedForward(nn.Module):
    def __init__(self, dim, hidden_dim, dropout = 0.):
        super().__init__()
        self.net = nn.Sequential(
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

        self.attend = nn.Softmax(dim = -1)
        self.to_qkv = nn.Linear(dim, inner_dim * 3, bias = False)

        self.to_out = nn.Sequential(
            nn.Linear(inner_dim, dim),
            nn.Dropout(dropout)
        ) if project_out else nn.Identity()

    def forward(self, x):
        qkv = self.to_qkv(x).chunk(3, dim = -1)
        q, k, v = map(lambda t: rearrange(t, 'b n (h d) -> b h n d', h = self.heads), qkv)

        dots = torch.matmul(q, k.transpose(-1, -2)) * self.scale

        attn = self.attend(dots)

        out = torch.matmul(attn, v)
        out = rearrange(out, 'b h n d -> b n (h d)')
        return self.to_out(out)

class Transformer(nn.Module):
    def __init__(self, dim, depth, heads, dim_head, mlp_dim, dropout = 0.):
        super().__init__()
        self.layers = nn.ModuleList([])
        for _ in range(depth):
            self.layers.append(nn.ModuleList([
                PreNorm(dim, Attention(dim, heads = heads, dim_head = dim_head, dropout = dropout)),
                PreNorm(dim, FeedForward(dim, mlp_dim, dropout = dropout))
            ]))
    def forward(self, x):
        for attn, ff in self.layers:
            x = attn(x) + x
            x = ff(x) + x
        return x


class Model(nn.Module):
    def __init__(self, *, image_size, patch_size, num_classes, dim, depth, heads, mlp_dim, pool = 'cls', channels = 3, dim_head = 64, dropout = 0., emb_dropout = 0.):
        super().__init__()
        image_height, image_width = pair(image_size)
        patch_height, patch_width = pair(patch_size)

        assert image_height % patch_height == 0 and image_width % patch_width == 0, 'Image dimensions must be divisible by the patch size.'

        num_patches = (image_height // patch_height) * (image_width // patch_width)
        patch_dim = channels * patch_height * patch_width
        assert pool in {'cls', 'mean'}, 'pool type must be either cls (cls token) or mean (mean pooling)'

        self.to_patch_embedding = nn.Sequential(
            Rearrange('b c (w p2) -> b w (p2 c)', p2 = patch_width),
            nn.Linear(patch_dim, dim),
        )

        self.pos_embedding = nn.Parameter(torch.randn(1, num_patches + 1, dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, dim))
        self.dropout = nn.Dropout(emb_dropout)

        self.transformer = Transformer(dim, depth, heads, dim_head, mlp_dim, dropout)

        self.pool = pool
        self.to_latent = nn.Identity()

        self.mlp_head = nn.Sequential(
            nn.LayerNorm(dim),
            nn.Linear(dim, num_classes)
        ) 
        ###########
        # self.vit = ViT(image_size=(1, 4000),
        #                patch_size=(1, 200),
        #                num_classes=5,
        #                dim=1024,
        #                depth=6, # 
        #                channels=7,
        #                heads=16,
        #                mlp_dim=2048,
        #                dropout=0.3, # 
        #                emb_dropout=0.1)
        self.fc = nn.Linear(7168, dim)
        self.DP = nn.Dropout(0.5)
        self.cnn1 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=64, padding=(10,),
                                            kernel_size=(50,), stride=(6,)),
                                  nn.MaxPool1d(kernel_size=8, stride=8),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Conv1d(64, 128, (8,), (1,), padding=(4,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (8,), (1,), padding=(4,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (8,), (1,), padding=(4,)), nn.ReLU(),
                                  nn.MaxPool1d(4, 4))
        self.cnn2 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=64, padding=(10,),
                                            kernel_size=(200,), stride=(16,)),
                                  nn.MaxPool1d(kernel_size=6, stride=6),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Conv1d(64, 128, (7,), (1,), padding=(4,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (7,), (1,), padding=(4,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (7,), (1,), padding=(4,)), nn.ReLU(),
                                  nn.MaxPool1d(3, 3))
        self.cnn3 = nn.Sequential(nn.Conv1d(in_channels=3, out_channels=64, padding=(10,),
                                            kernel_size=(400,), stride=(50,)),
                                  nn.MaxPool1d(kernel_size=4, stride=4),
                                  nn.ReLU(),
                                  nn.Dropout(0.3),
                                  nn.Conv1d(64, 128, (6,), (1,), padding=(3,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (6,), (1,), padding=(3,)), nn.ReLU(),
                                  nn.Conv1d(128, 128, (6,), (1,), padding=(3,)), nn.ReLU(),
                                  nn.MaxPool1d(2, 2))
        self.lstm = nn.LSTM(input_size=3400, hidden_size=1024, num_layers=2)
        self.gelu = nn.GELU()
        #############

    def forward(self, img):
        #############
        x = self.to_patch_embedding(img[:, :, 1000:4000])
        b, n, _ = x.shape

        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b = b)
        x = torch.cat((cls_tokens, x), dim=1)
        x += self.pos_embedding[:, :(n + 1)]
        x = self.dropout(x)

        x = self.transformer(x)

        x = x.mean(dim = 1) if self.pool == 'mean' else x[:, 0]
        ########################
        img_prev = img[:, :, :2000]
        prev_1 = self.cnn1(img_prev).view(img_prev.shape[0], -1)
        prev_2 = self.cnn2(img_prev).view(img_prev.shape[0], -1)
        prev_3 = self.cnn3(img_prev).view(img_prev.shape[0], -1)
        img_prev = torch.cat((prev_1, prev_2, prev_3), dim=1)

        img_behind = img[:, :, 3000:]
        behind_1 = self.cnn1(img_behind).view(img_behind.shape[0], -1)
        behind_2 = self.cnn2(img_behind).view(img_behind.shape[0], -1)
        behind_3 = self.cnn3(img_behind).view(img_behind.shape[0], -1)
        img_behind = torch.cat((behind_1, behind_2, behind_3), dim=1)

        x = torch.cat((img_prev, x, img_behind), dim=1)
        x = self.gelu(self.fc(x))
        ##########
        x = self.to_latent(x)
        return self.mlp_head(x)
