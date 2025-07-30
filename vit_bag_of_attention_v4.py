import os
import zipfile
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader, random_split
from torchvision import transforms
from PIL import Image
import torch.nn.functional as F
from torch.optim.lr_scheduler import OneCycleLR

# Extract dataset
# zip_path = "train_and_validation_set.zip"
# extract_path = "."

# with zipfile.ZipFile(zip_path, 'r') as zip_ref:
    # zip_ref.extractall(extract_path)

# print(f"Dataset extracted to {extract_path}")

# Custom Dataset class
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

        return image, label

# Define transforms
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
batch_size = 8
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")

# Model hyperparameters
learning_rate = 0.0001
weight_decay = 0.0001
num_epochs = 300
image_size = 512  # We'll resize input images to this size
patch_size = 16  # Size of the patches to be extract from the input images
num_patches = (image_size // patch_size) ** 2
projection_dim = 256
num_heads = 8
transformer_units = [
    projection_dim * 2,
    projection_dim,
]  # Size of the transformer layers
transformer_layers = 8
mlp_head_units = [
    2048,
    1024,
]  # Size of the dense layers of the final classifier
num_classes = 4  # Mayo 0, 1, 2, 3

# FFT Feature Extraction
class FFTFeatureExtraction(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, x):
        # Apply FFT to the input images
        # PyTorch expects x in shape [batch, channels, height, width]
        x_complex = torch.fft.fft2(x.float())
        magnitude = torch.abs(x_complex)
        return magnitude

# Channel Attention Module
class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)

        self.shared_mlp = nn.Sequential(
            nn.Conv2d(in_channels, in_channels // reduction_ratio, kernel_size=1, bias=False),
            nn.ReLU(),
            nn.Conv2d(in_channels // reduction_ratio, in_channels, kernel_size=1, bias=False)
        )

        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch, channels, height, width]
        avg_out = self.shared_mlp(self.avg_pool(x))
        max_out = self.shared_mlp(self.max_pool(x))
        out = self.sigmoid(avg_out + max_out)
        return x * out

# Spatial Attention Module
class SpatialAttention(nn.Module):
    def __init__(self, kernel_size=7):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(2, 1, kernel_size=kernel_size, padding=kernel_size//2, bias=False)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        # x shape: [batch, channels, height, width]
        avg_out = torch.mean(x, dim=1, keepdim=True)
        max_out, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avg_out, max_out], dim=1)
        out = self.conv(out)

        return x * self.sigmoid(out)

# Combine attention modules
class BagOfAttentions(nn.Module):
    def __init__(self, embed_dim, num_heads=4, dropout=0.1):
        super(BagOfAttentions, self).__init__()
        # Keep the original multi-head attention for transformer sequences
        self.mha = nn.MultiheadAttention(
            embed_dim=embed_dim,
            num_heads=num_heads,
            dropout=dropout,
            batch_first=True
        )

        # Channel attention adapter for sequence data
        # For sequence data of shape [batch, seq_len, embed_dim]
        # We'll reshape to [batch, embed_dim, seq_len] for channel attention
        self.channel_attn = nn.Sequential(
            nn.Linear(embed_dim, embed_dim // 8),
            nn.ReLU(),
            nn.Linear(embed_dim // 8, embed_dim),
            nn.Sigmoid()
        )

        # Spatial attention adapter for sequence data
        # This operates on the sequence dimension
        self.spatial_attn = nn.Sequential(
            nn.Linear(embed_dim, 1),
            nn.Sigmoid()
        )

        # Learnable weights for combining attention types
        self.weights = nn.Parameter(torch.ones(3))  # [mha, channel, spatial]

    def forward(self, x):
        # Input x is [batch, seq_len, embed_dim]
        batch_size, seq_len, embed_dim = x.shape

        # 1. Multi-head attention
        mha_out, _ = self.mha(x, x, x)  # Standard self-attention

        # 2. Channel attention (across embed_dim)
        # Calculate channel attention weights
        # First, compute mean across sequence dimension
        avg_pool = torch.mean(x, dim=1, keepdim=True)  # [batch, 1, embed_dim]
        max_pool, _ = torch.max(x, dim=1, keepdim=True)  # [batch, 1, embed_dim]

        # Pass through channel attention MLP
        avg_out = self.channel_attn(avg_pool.squeeze(1)).unsqueeze(1)  # [batch, 1, embed_dim]
        max_out = self.channel_attn(max_pool.squeeze(1)).unsqueeze(1)  # [batch, 1, embed_dim]

        # Combine and apply channel attention
        channel_weights = avg_out + max_out  # [batch, 1, embed_dim]
        channel_out = x * channel_weights  # [batch, seq_len, embed_dim]

        # 3. Spatial attention (across sequence dimension)
        # Get spatial attention weights
        spatial_weights = self.spatial_attn(x)  # [batch, seq_len, 1]
        spatial_out = x * spatial_weights  # [batch, seq_len, embed_dim]

        # Normalize weights for combining attention types
        weights = F.softmax(self.weights, dim=0)

        # Combine the three attention outputs
        combined_out = (
            weights[0] * mha_out +
            weights[1] * channel_out +
            weights[2] * spatial_out
        )

        return combined_out

# Patches extraction
class PatchEmbedding(nn.Module):
    def __init__(self, img_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.n_patches = (img_size // patch_size) ** 2

        # Convert patches to embedding vectors
        self.proj = nn.Conv2d(
            in_channels, embed_dim,
            kernel_size=patch_size, stride=patch_size
        )

        # Position embeddings for patches + class token
        self.position_embedding = nn.Parameter(torch.zeros(1, self.n_patches + 1, embed_dim))
        torch.nn.init.trunc_normal_(self.position_embedding, std=0.02)

    def forward(self, x):
        # x: [B, C, H, W]
        batch_size = x.shape[0]

        # Extract patches with convolution
        x = self.proj(x)  # [B, embed_dim, H/patch_size, W/patch_size]

        # Flatten patches to sequence
        x = x.flatten(2)  # [B, embed_dim, n_patches]
        x = x.transpose(1, 2)  # [B, n_patches, embed_dim]

        # Add position embeddings
        # x = x + self.position_embedding

        return x

# MLP block for Transformer
class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, dropout=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.dropout(x)
        x = self.fc2(x)
        x = self.dropout(x)
        return x


# Transformer Encoder Block with Bag of Attentions
class TransformerBlock(nn.Module):
    def __init__(self, dim, num_heads, mlp_ratio=4, dropout=0.1):
        super().__init__()
        # Layer normalization
        self.norm1 = nn.LayerNorm(dim)

        # Bag of Attentions instead of just Multi-head attention
        self.attn = BagOfAttentions(
            embed_dim=dim,
            num_heads=num_heads,
            dropout=dropout
        )

        # Layer normalization
        self.norm2 = nn.LayerNorm(dim)

        # MLP
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = MLP(
            in_features=dim,
            hidden_features=mlp_hidden_dim,
            out_features=dim,
            dropout=dropout
        )

    def forward(self, x):
        # Apply attention with residual connection
        x_norm = self.norm1(x)
        attn_output = self.attn(x_norm)
        # print('attn',attn_output.shape)  # Commented out for performance
        x = x + attn_output

        # Apply MLP with residual connection
        x = x + self.mlp(self.norm2(x))
        # print('x_with_res_conn', x.shape)  # Commented out for performance
        return x

# Complete Vision Transformer
class VisionTransformer(nn.Module):
    def __init__(
        self,
        img_size=512,
        patch_size=16,
        in_channels=3,
        num_classes=4,
        embed_dim=64,
        depth=8,
        num_heads=4,
        mlp_ratio=4,
        dropout=0.1
    ):
        super().__init__()

        # FFT feature extraction
        self.fft = FFTFeatureExtraction()

        self.depth = depth

        # Patch embedding
        self.patch_embed = PatchEmbedding(
            img_size=img_size,
            patch_size=patch_size,
            in_channels=in_channels,
            embed_dim=embed_dim
        )
        # Add class token
        self.cls_token = nn.Parameter(torch.zeros(1, 1, embed_dim))
        torch.nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.pos_drop = nn.Dropout(p=dropout)

        # Transformer blocks with Bag of Attentions
        self.transformer_blocks = nn.ModuleList([
            TransformerBlock(
                dim=embed_dim,
                num_heads=num_heads,
                mlp_ratio=mlp_ratio,
                dropout=dropout
            )
            for _ in range(depth)
        ])


          # Skip connection projection layers
        # These will help match dimensions if needed
        self.skip_projections = nn.ModuleList([
            nn.Identity()  # For simplicity, using Identity if dimensions match
            for _ in range(depth//2)
        ])

        # Layer normalization
        self.norm = nn.LayerNorm(embed_dim)

        # MLP Head for classification
        self.head = nn.Sequential(
            nn.Linear(embed_dim, mlp_head_units[0]),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_head_units[0], mlp_head_units[1]),
            nn.GELU(),
            nn.Dropout(0.5),
            nn.Linear(mlp_head_units[1], num_classes)
        )
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, nn.Linear):
            torch.nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                torch.nn.init.zeros_(m.bias)
        elif isinstance(m, nn.LayerNorm):
            torch.nn.init.ones_(m.weight)
            torch.nn.init.zeros_(m.bias)

    def forward(self, x):
        # print('x.shape', x.shape)  # Commented out for performance

        # Apply FFT
        # x = self.fft(x)

        # print('fft.shape', x.shape)  # Commented out for performance

        # Extract patches and embed
        x = self.patch_embed(x)
        # print('patchembd.shape', x.shape)  # Commented out for performance

        # Add class token
        batch_size = x.shape[0]
        cls_tokens = self.cls_token.expand(batch_size, -1, -1)
        x = torch.cat((cls_tokens, x), dim=1)
        

         # Add position embeddings
        x = x + self.patch_embed.position_embedding
        x = self.pos_drop(x)

        # Store intermediate outputs for skip connections
        skip_outputs = []
        # Apply first half of transformer blocks
        for i in range(self.depth//2):
            x = self.transformer_blocks[i](x)
            skip_outputs.append(x)

        # Apply second half of transformer blocks with skip connections
        for i in range(self.depth//2, self.depth):
            # Get corresponding skip connection from earlier layer
            skip_idx = self.depth - i - 1
            skip_connection = self.skip_projections[skip_idx](skip_outputs[skip_idx])
            
            # Apply transformer block
            x = self.transformer_blocks[i](x)
            
            # Add skip connection (residual connection across blocks)
            x = x + skip_connection
        
        # Apply transformer blocks
        # for block in self.transformer_blocks:
        #     x = block(x)
        # print('transformed.shape', x.shape)  # Commented out for performance
        
        # Global average pooling
        x = self.norm(x)
        x = x[:, 0]  # [B, embed_dim]
        # print('globalavgpool.shape', x.shape)  # Commented out for performance
        
        # Classification head
        x = self.head(x)
        # print('classficationhead.shape', x.shape)  # Commented out for performance
        
        return x

# Initialize model
model = VisionTransformer(
    img_size=image_size,
    patch_size=patch_size,
    in_channels=3,
    num_classes=num_classes,
    embed_dim=projection_dim,
    depth=transformer_layers,
    num_heads=num_heads,
    mlp_ratio=4,
    dropout=0.1
)

# Move model to GPU if available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = model.to(device)
print(f"Using device: {device}")


# Loss function and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Replace ReduceLROnPlateau with CosineAnnealingLR
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)


# Training function
def train_model(model, train_loader, val_loader, criterion, optimizer, num_epochs, log_file_path='training_log.txt'):
    best_val_acc = 0.0

    with open(log_file_path, 'a') as log_file:
        for epoch in range(num_epochs):
            print(f"Training for epoch: {epoch}")
            log_file.write(f"Training for epoch: {epoch}\n")
            # Training phase
            model.train()
            running_loss = 0.0
            correct = 0
            total = 0

            for images, labels in train_loader:
                images, labels = images.to(device), labels.to(device)

                # Zero the parameter gradients
                optimizer.zero_grad()

                # Forward pass
                outputs = model(images)
                loss = criterion(outputs, labels)

                # Backward pass and optimize
                loss.backward()
                # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                optimizer.step()

                # Statistics
                running_loss += loss.item() * images.size(0)
                _, predicted = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()

            epoch_loss = running_loss / len(train_loader.dataset)
            epoch_acc = correct / total

            # Validation phase
            model.eval()
            val_loss = 0.0
            val_correct = 0
            val_total = 0

            with torch.no_grad():
                for images, labels in val_loader:
                    images, labels = images.to(device), labels.to(device)

                    # Forward pass
                    outputs = model(images)
                    loss = criterion(outputs, labels)

                    # Statistics
                    val_loss += loss.item() * images.size(0)
                    _, predicted = torch.max(outputs, 1)
                    val_total += labels.size(0)
                    val_correct += (predicted == labels).sum().item()

            val_loss = val_loss / len(val_loader.dataset)
            val_acc = val_correct / val_total

            # Save the best model
            if val_acc > best_val_acc:
                best_val_acc = val_acc
                torch.save(model.state_dict(), 'best_model.pth')
            
            scheduler.step()
            print(f'Epoch {epoch+1}/{num_epochs}:')
            print(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}')
            print(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}')
            log_file.write(f'Epoch {epoch+1}/{num_epochs}:\n')
            log_file.write(f'Train Loss: {epoch_loss:.4f}, Train Acc: {epoch_acc:.4f}\n')
            log_file.write(f'Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.4f}\n')
  
    return model


# Evaluate function
def evaluate_model(model, test_loader, log_file_path='training_log.txt'):
    model.eval()
    correct = 0
    total = 0

    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)

            # Forward pass
            outputs = model(images)
            _, predicted = torch.max(outputs, 1)

            # Statistics
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    
    test_acc = correct / total
    
    with open(log_file_path, 'a') as log_file:
        log_file.write(f'Test Accuracy: {test_acc:.4f}\n')
        
        
    print(f'Test Accuracy: {test_acc:.4f}')

    return test_acc


# Run the training
print("Starting training...")
model = train_model(model, train_dataloader, val_dataloader, criterion, optimizer, num_epochs, 'vit_bag_of_attention_v4.txt')

# Evaluate the model
print("Evaluating model...")
test_acc = evaluate_model(model, test_dataloader, 'vit_bag_of_attention_v4.txt')
