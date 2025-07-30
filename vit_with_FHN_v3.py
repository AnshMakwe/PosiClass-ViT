import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
import torchvision
from torchvision import transforms
from PIL import Image
from einops import rearrange
from einops.layers.torch import Rearrange
import torch.nn.functional as F

# Dataset class - keeping this part similar to original
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.classes = {"Mayo 0": 0, "Mayo 1": 1, "Mayo 2": 2, "Mayo 3": 3}  # Assign labels

        # Collect all images with labels
        self.image_paths = []
        for class_name, label in self.classes.items():
            class_dir = os.path.join(root_dir, class_name)
            for img_file in os.listdir(class_dir):
                if img_file.endswith(('.png', '.jpg', '.jpeg', '.bmp')):  # Image formats
                    self.image_paths.append((os.path.join(class_dir, img_file), label))

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path, label = self.image_paths[idx]
        image = Image.open(img_path).convert("RGB")

        if self.transform:
            image = self.transform(image)

        return image, label  # Return image tensor and label

# PyTorch Vision Transformer Implementation
class PatchEmbed(nn.Module):
    def __init__(self, img_size=224, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        self.proj = nn.Sequential(
            # Convert image into patches and flatten
            Rearrange('b c (h p1) (w p2) -> b (h w) (p1 p2 c)',
                      p1=patch_size, p2=patch_size),
            nn.Linear(patch_size * patch_size * in_chans, embed_dim)
        )

    def forward(self, x):
        return self.proj(x)

class Attention(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class FHNActivation(nn.Module):
    def __init__(self, a=0.7, b=0.8, I_ext_min=-1.5, I_ext_max=1.5, num_points=100):
        super().__init__()
        self.a = a
        self.b = b
        self.register_buffer('I_ext_values', torch.linspace(I_ext_min, I_ext_max, num_points))
        # Precompute v values using Newton-Raphson method
        self.register_buffer('v_values', self._compute_v_values())
        
    def _equation(self, v, I_ext):
        # The FHN cubic equation
        return v - (v ** 3) / 3 - (v + self.a) / self.b + I_ext
    
    def _compute_v_values(self):
        # Use Newton-Raphson method to solve for v for each I_ext value
        v_values = []
        for I in self.I_ext_values:
            # Start with initial guess v=0
            v = torch.tensor(0.0, requires_grad=True, device=self.I_ext_values.device)
            # Newton-Raphson iterations
            for _ in range(50):  # Usually converges in fewer iterations
                f = self._equation(v, I)
                f.backward()  # Compute gradient (derivative)
                with torch.no_grad():
                    v -= f / v.grad  # Newton step: v = v - f(v)/f'(v)
                v.grad.zero_()  # Reset gradient for next iteration
            v_values.append(v.detach())  # Store converged value
        return torch.stack(v_values)
    
    def forward(self, x):
        # Ensure tensors are on the same device
        device = x.device
        
        # Clamp input to interpolation range
        x_clamped = torch.clamp(x, min=self.I_ext_values[0], max=self.I_ext_values[-1])
        
        # Find indices for linear interpolation
        idx = torch.bucketize(x_clamped, self.I_ext_values) - 1
        idx = torch.clamp(idx, 0, len(self.I_ext_values) - 2)
        
        # Get neighboring points for interpolation
        x0 = self.I_ext_values[idx]
        x1 = self.I_ext_values[idx + 1]
        y0 = self.v_values[idx]
        y1 = self.v_values[idx + 1]
        
        # Linear interpolation: y = y0 + (x - x0) * (y1 - y0) / (x1 - x0)
        slope = (y1 - y0) / (x1 - x0)
        y = y0 + slope * (x_clamped - x0)
        
        return y


class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = FHNActivation() 
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x

class Block(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0., attn_drop=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = BagOfAttentions(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    

class BagOfAttentions(nn.Module):
    def __init__(self, dim, num_heads=8, qkv_bias=False, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.dim = dim
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = head_dim ** -0.5

        # Original Multi-Head Self-Attention
        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)

        # Channel Attention
        self.channel_attn = ChannelAttention(dim)

        # Spatial Attention
        self.spatial_attn = SpatialAttention()

        # Fusion layer
        self.fusion = nn.Parameter(torch.ones(3) / 3)  # Equal weights initially
        self.softmax = nn.Softmax(dim=0)

        # Output projection
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x):
        B, N, C = x.shape

        # 1. Multi-Head Self-Attention (MHSA)
        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        mhsa_out = (attn @ v).transpose(1, 2).reshape(B, N, C)

        # 2. Channel Attention
        channel_out = self.channel_attn(x)

        # 3. Spatial Attention
        spatial_out = self.spatial_attn(x)

        # Combine all attention outputs with learnable weights
        fusion_weights = self.softmax(self.fusion)
        x = fusion_weights[0] * mhsa_out + fusion_weights[1] * channel_out + fusion_weights[2] * spatial_out

        # Final projection
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class ChannelAttention(nn.Module):
    def __init__(self, dim, reduction_ratio=16):
        super().__init__()
        self.avg_pool = nn.AdaptiveAvgPool1d(1)
        self.max_pool = nn.AdaptiveMaxPool1d(1)

        # Shared MLP
        self.mlp = nn.Sequential(
            nn.Linear(dim, dim // reduction_ratio, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction_ratio, dim, bias=False)
        )
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, N, C = x.shape

        # Transpose for pooling along sequence dimension
        x_trans = x.transpose(1, 2)  # B, C, N

        # Average pooling
        avg_out = self.avg_pool(x_trans).view(B, C)
        avg_out = self.mlp(avg_out).view(B, C, 1)

        # Max pooling
        max_out = self.max_pool(x_trans).view(B, C)
        max_out = self.mlp(max_out).view(B, C, 1)

        # Attention weights
        attn = self.sigmoid(avg_out + max_out)  # B, C, 1

        # Apply weights to each channel
        out = x * attn.view(B, 1, C)

        return out


class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super().__init__()
        padding = kernel_size // 2
        self.conv = nn.Conv1d(2, 1, kernel_size, padding=padding, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        B, N, C = x.shape

        # Calculate average and max features across channels
        avg_out = torch.mean(x, dim=2, keepdim=True)
        max_out, _ = torch.max(x, dim=2, keepdim=True)

        # Concatenate along channel dimension
        attn = torch.cat([avg_out, max_out], dim=2)  # B, N, 2
        attn = attn.transpose(1, 2)  # B, 2, N

        # Apply convolution
        attn = self.conv(attn)  # B, 1, N
        attn = self.sigmoid(attn)  # B, 1, N

        # Apply spatial attention weights
        out = x * attn.transpose(1, 2)  # B, N, C

        return out


class VisionTransformer(nn.Module):
    def __init__(self, img_size=72, patch_size=6, in_chans=3, num_classes=4,
                 embed_dim=64, depth=8, num_heads=4, mlp_ratio=2.,
                 qkv_bias=True, drop_rate=0.1, attn_drop_rate=0.1):
        super().__init__()
        self.num_classes = num_classes
        self.num_features = embed_dim
        self.embed_dim = embed_dim

        # Patch embedding
        self.patch_embed = PatchEmbed(img_size=img_size, patch_size=patch_size,
                                     in_chans=in_chans, embed_dim=embed_dim)
        num_patches = self.patch_embed.n_patches

        # Class token and position embeddings
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.zeros(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # MLP Head
        self.head = nn.Sequential(
            nn.Linear(embed_dim, 2048),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(2048, 1024),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(1024, num_classes)
        )

        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):
        B = x.shape[0]
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)

        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        x = torch.cat((cls_token, x), dim=1)  # (B, 1 + n_patches, embed_dim)

        # Add position embedding
        x = x + self.pos_embed  # (B, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)

        # Apply Transformer blocks
        x = self.blocks(x)
        x = self.norm(x)

        # Extract cls token representation
        return x[:, 0]

    def forward(self, x):
        x = self.forward_features(x)
        x = self.head(x)
        return x

# Setup transformations
train_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.RandomRotation(30),
    transforms.RandomHorizontalFlip(),
    transforms.RandomVerticalFlip(),
    transforms.RandomAffine(degrees=10, scale=(0.7, 1.3), shear=15), 
    transforms.ToTensor(),
    transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create train and validation datasets from train_and_validation_set directory
train_val_dataset = ImageDataset(root_dir="/home/ubuntu/train_and_validation_set", transform=train_transform)

# Create test dataset from test_set directory
test_dataset = ImageDataset(root_dir="/home/ubuntu/test_set", transform=test_transform)

# Compute train and validation split sizes (70/30 split)
total_size = len(train_val_dataset)
train_size = int(0.7 * total_size)
val_size = total_size - train_size

# Split train_val_dataset into train and validation datasets
train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])





# Create DataLoaders
batch_size = 8  # Smaller batch size for memory efficiency
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training parameters
num_classes = len(os.listdir("train_and_validation_set"))
learning_rate = 0.0001
weight_decay = 0.0001
num_epochs = 1

# Initialize model
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = VisionTransformer(
    img_size=512,
    patch_size=16,
    in_chans=3,
    num_classes=num_classes,
    embed_dim=64,
    depth=8,
    num_heads=4
).to(device)

# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)

# Training function
def train_one_epoch(model, dataloader, criterion, optimizer, device):
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0

    for inputs, labels in dataloader:
        inputs, labels = inputs.to(device), labels.to(device)

        # Zero the parameter gradients
        optimizer.zero_grad()

        # Forward pass
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        # Backward pass and optimize
        loss.backward()
        optimizer.step()

        # Statistics
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total

    return epoch_loss, epoch_acc

# Validation function with class-wise accuracy
def validate(model, dataloader, criterion, device, log_file_path):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Add class-wise tracking
    class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    class_total = {0: 0, 1: 0, 2: 0, 3: 0}
    class_names = ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3']

    with torch.no_grad():
        for inputs, labels in dataloader:
            inputs, labels = inputs.to(device), labels.to(device)

            # Forward pass
            outputs = model(inputs)
            loss = criterion(outputs, labels)

            # Statistics
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
            
            # Class-wise statistics
            for label, prediction in zip(labels, predicted):
                label_idx = label.item()
                class_total[label_idx] += 1
                if label == prediction:
                    class_correct[label_idx] += 1

    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'Test Accuracy: {epoch_acc:.4f}\n')
        # Add class-wise accuracy reporting
        for cls_idx, cls_name in enumerate(class_names):
            if class_total[cls_idx] > 0:
                acc = class_correct[cls_idx] / class_total[cls_idx]
            else:
                acc = 0.0
            log_file.write(f'Class {cls_name} Accuracy: {acc:.4f}\n')

    return epoch_loss, epoch_acc


# Training loop
def train_model(model, train_loader, val_loader, criterion, optimizer, scheduler, num_epochs, device, log_file_path):
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    with open(log_file_path, 'a') as log_file:
        for epoch in range(num_epochs):
            print(f"Training for epoch: {epoch}")
            log_file.write(f"Training for epoch: {epoch}\n")
        
            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_loss, val_acc = validate(model, val_loader, criterion, device, log_file_path)

            # Update scheduler
            scheduler.step()

            # Save history
            history['train_loss'].append(train_loss)
            history['train_acc'].append(train_acc)
            history['val_loss'].append(val_loss)
            history['val_acc'].append(val_acc)

            # Save best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')

            print(f'Epoch {epoch+1}/{num_epochs} | '
                f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}')
            
            print(f'Epoch {epoch+1}/{num_epochs}:\n')
            print(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n')
            log_file.write(f'Epoch {epoch+1}/{num_epochs}:\n')
            log_file.write(f'Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.4f}\n')
            log_file.write(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n')

    return history

# Run the training
history = train_model(
    model, train_dataloader, val_dataloader,
    criterion, optimizer, scheduler,
    num_epochs, device, "vit_with_FHN_v3.txt"
)

# Load best model for evaluation
model.load_state_dict(torch.load('best_model.pth'))

# Evaluate on test set
test_loss, test_acc = validate(model, test_dataloader, criterion, device, "vit_with_FHN_v3.txt")
print(f'Test Loss: {test_loss:.4f} | Test Acc: {test_acc:.4f}')

