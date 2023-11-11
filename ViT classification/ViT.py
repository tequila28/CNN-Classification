from torch import nn
import torch
class ViT(nn.Module):
    def __init__(self, image_size, patch_size, num_classes, hidden_dim, num_heads, num_layers):
        super(ViT, self).__init__()
        self.patch_embedding = nn.Conv2d(3, hidden_dim, kernel_size=patch_size, stride=patch_size)
        num_patches = (image_size // patch_size) ** 2
        self.positional_embedding = nn.Parameter(torch.randn(num_patches, hidden_dim))
        self.transformer = nn.Transformer(hidden_dim, num_heads, num_layers)
        self.fc1 = nn.Sequential(nn.Linear(64, 1),nn.ReLU())
        self.fc2 = nn.Sequential(nn.Linear(32, 10))

    def forward(self, x):
        x = self.patch_embedding(x)
        x = x.permute(0, 2, 3, 1).view(x.size(0), -1, x.size(1))
        x = x + self.positional_embedding
        x = x.permute(1, 0, 2)
        x = self.transformer(x, x)
        x = x.permute(1, 2, 0)
        x=self.fc1(x)
        x = self.fc2(x.squeeze(2))
        return x
