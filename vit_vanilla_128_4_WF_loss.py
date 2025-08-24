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
    def __init__(self, img_size=512, patch_size=16, in_chans=3, embed_dim=768):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        self.hidden_size = embed_dim
        self.num_channels = in_chans
        self.n_patches = (img_size // patch_size) ** 2
        

        self.proj = nn.Conv2d(self.num_channels, self.hidden_size, kernel_size=self.patch_size, stride=self.patch_size)
    def forward(self, x):
        x = self.proj(x)
        x = x.flatten(2).transpose(1, 2)
        # print("patch embedding: ", x.shape)
        return x

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

class MLP(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, drop=0.1):
        super().__init__()
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = nn.GELU()
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
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False, drop=0.1, attn_drop=0.1):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim, eps=1e-6)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias, attn_drop=attn_drop, proj_drop=drop)
        self.norm2 = nn.LayerNorm(dim, eps=1e-6)
        self.mlp = MLP(in_features=dim, hidden_features=int(dim * mlp_ratio), out_features=dim, drop=drop)

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x
    


class VisionTransformer(nn.Module):
    def __init__(self, img_size=512, patch_size=16, in_chans=3, num_classes=4,
                 embed_dim=768, depth=8, num_heads=4, mlp_ratio=2.,
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
        self.cls_token = nn.Parameter(torch.rand(1, 1, embed_dim))
        self.pos_embed = nn.Parameter(torch.rand(1, num_patches + 1, embed_dim))
        self.pos_drop = nn.Dropout(p=drop_rate)

        # Transformer blocks
        self.blocks = nn.Sequential(*[
            Block(dim=embed_dim, num_heads=num_heads, mlp_ratio=mlp_ratio,
                  qkv_bias=qkv_bias, drop=drop_rate, attn_drop=attn_drop_rate)
            for _ in range(depth)
        ])
        self.norm = nn.LayerNorm(embed_dim, eps=1e-6)

        # MLP Head
        # self.head = nn.Sequential(
        #     nn.Linear(embed_dim, 2048),
        #     nn.GELU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(2048, 1024),
        #     nn.GELU(),
        #     nn.Dropout(0.5),
        #     nn.Linear(1024, num_classes)
        # )

        # Standard ViT head implementation
        self.head = nn.Linear(embed_dim, num_classes)


        # Initialize weights
        nn.init.trunc_normal_(self.pos_embed, std=0.02)
        nn.init.trunc_normal_(self.cls_token, std=0.02)
        self.apply(self._init_weights)

    def _init_weights(self, m):
        if isinstance(m, (nn.Linear, nn.Conv2d)):
            nn.init.trunc_normal_(m.weight, std=0.02)
            if m.bias is not None:
                nn.init.constant_(m.bias, 0)
        elif isinstance(m, nn.LayerNorm):
            nn.init.constant_(m.bias, 0)
            nn.init.constant_(m.weight, 1.0)

    def forward_features(self, x):

        B = x.shape[0]
        # print("--------------------------------------")
        # print("x shape: ", x.shape)
        # print("--------------------------------------")
        x = self.patch_embed(x)  # (B, n_patches, embed_dim)
        # print("x patch embedding shape: ", x.shape)
        # print("--------------------------------------")


        # Add class token
        cls_token = self.cls_token.expand(B, -1, -1)  # (B, 1, embed_dim)
        # print("clas token shape: ", cls_token.shape)
        # print("printing cls tokens")
        # print(cls_token)
        # print("--------------------------------------")
        x = torch.cat((cls_token, x), dim=1)  # (B, 1 + n_patches, embed_dim)
        # print("x conactenation cls token shape: ", x.shape)
        # print("--------------------------------------")
        # print("printing x after cat cls token: ")
        # print(x)
        # print("--------------------------------------")

        # Add position embedding
        x = x + self.pos_embed  # (B, 1 + n_patches, embed_dim)
        x = self.pos_drop(x)
        # print("--------------------------------------")
        # print("x  adding pos embeding shape: ", x.shape)
        # print("--------------------------------------")
        # print("printing x after adding position embediing: ")
        # print(x)
        # print("--------------------------------------")

        # Apply Transformer blocks
        x = self.blocks(x)
        # print("x after block shape: ", x.shape)
        
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
    # transforms.RandomErasing(p=0.5, scale=(0.02, 0.33), ratio=(0.3, 3.3), value=0),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

test_transform = transforms.Compose([
    transforms.Resize((512, 512)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

# Create train and validation datasets from train_and_validation_set directory
train_dataset = ImageDataset(root_dir="/home/user/Ansh/train_and_validation_sets/train/", transform=train_transform)


val_dataset = ImageDataset(root_dir="/home/user/Ansh/train_and_validation_sets/val/", transform=train_transform)


# Create test dataset from test_set directory
test_dataset = ImageDataset(root_dir="/home/user/Ansh/test_set/", transform=test_transform)

# Compute train and validation split sizes (70/30 split)
# total_size = len(train_val_dataset)
train_size = len(train_dataset)
val_size = len(val_dataset)

# Split train_val_dataset into train and validation datasets
# train_dataset, val_dataset = random_split(train_val_dataset, [train_size, val_size])

print(f"Train dataset size: {len(train_dataset)}")
print(f"Validation dataset size: {len(val_dataset)}")
print(f"Test dataset size: {len(test_dataset)}")


# Create DataLoaders
batch_size = 32
train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=batch_size, shuffle=False)
test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False)

# Training parameters
num_classes = 4
learning_rate = 0.0001
weight_decay = 0.0001
num_epochs = 200

# Initialize model
device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
print(f"Using device: {device}")

model = VisionTransformer(
    img_size=512,
    patch_size=16,
    in_chans=3,
    num_classes=num_classes,
    embed_dim=128,
    depth=8,
    num_heads=4
).to(device)


# Weber-Fechner Loss function
# class WeberFechnerLoss(nn.Module):
#     def __init__(self, base_criterion=nn.CrossEntropyLoss(), weber_weight=0.3):
#         super().__init__()
#         self.base_criterion = base_criterion
#         self.weber_weight = weber_weight
        
#     def forward(self, outputs, targets):
#         # Standard loss
#         base_loss = self.base_criterion(outputs, targets)
        
#         # Weber-Fechner component that penalizes according to perceptual differences
#         # Apply softmax to get probabilities
#         probs = F.softmax(outputs, dim=1)
#         # Apply log transformation to model perceptual differences
#         perceptual_probs = torch.log(probs + 1e-5)
#         # Calculate perceptual loss
#         perceptual_loss = F.nll_loss(perceptual_probs, targets)
        
#         # Combine losses
#         return (1 - self.weber_weight) * base_loss + self.weber_weight * perceptual_loss


class WeberFechnerLoss(nn.Module):
    def __init__(self, base_criterion=nn.CrossEntropyLoss(),num_classes=4, weber_weight=0.3):
        super().__init__()
        self.base_criterion = base_criterion
        self.weber_weight = weber_weight
        self.num_classes = num_classes
        
        # Create distance matrix for Mayo scores
        distance_matrix = torch.zeros(num_classes, num_classes)
        for i in range(num_classes):
            for j in range(num_classes):
                distance_matrix[i][j] = abs(i - j)
        self.register_buffer('distance_matrix', distance_matrix)
    
    def forward(self, outputs, targets):
        base_loss = self.base_criterion(outputs, targets)
        
        # Weber-Fechner component based on class distances
        probs = F.softmax(outputs, dim=1)
        
        # Calculate expected distance penalty
        distance_penalty = 0
        for i, target in enumerate(targets):
            for j in range(self.num_classes):
                # Logarithmic penalty based on distance (Weber-Fechner law)
                if self.distance_matrix[target][j] > 0:
                    penalty = torch.log(1 + self.distance_matrix[target][j])
                    distance_penalty += probs[i][j] * penalty
        
        distance_penalty = distance_penalty / len(targets)
        
        return base_loss + self.weber_weight * distance_penalty



# Loss function and optimizer
base_criterion = nn.CrossEntropyLoss()
criterion = WeberFechnerLoss(base_criterion=base_criterion, num_classes=num_classes, weber_weight=0.3)
optimizer = optim.AdamW(model.parameters(), lr=learning_rate, weight_decay=weight_decay)

# Learning rate scheduler
scheduler = optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=num_epochs)



# Training function for one epoch
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
def validate(model, dataloader, criterion, device, epoch, log_file_path=None, print_results=False):
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    # Add class-wise tracking
    class_correct = {0: 0, 1: 0, 2: 0, 3: 0}
    class_total = {0: 0, 1: 0, 2: 0, 3: 0}
    class_names = ['Mayo 0', 'Mayo 1', 'Mayo 2', 'Mayo 3']
    
    # Initialize confusion matrix
    num_classes = 4
    confusion_matrix = torch.zeros(num_classes, num_classes)
    
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
                pred_idx = prediction.item()
                class_total[label_idx] += 1
                confusion_matrix[label_idx][pred_idx] += 1
                if label == prediction:
                    class_correct[label_idx] += 1
    
    epoch_loss = running_loss / len(dataloader.dataset)
    epoch_acc = correct / total
    
    # Report results
    if log_file_path:
        with open(log_file_path, 'a') as log_file:
            log_file.write(f'Overall Accuracy {epoch}: {epoch_acc:.4f}\n')
            
            # Add class-wise accuracy reporting
            for cls_idx, cls_name in enumerate(class_names):
                if class_total[cls_idx] > 0:
                    acc = class_correct[cls_idx] / class_total[cls_idx]
                else:
                    acc = 0.0
                log_file.write(f'Class {cls_name} - Correct: {class_correct[cls_idx]}, Total: {class_total[cls_idx]}, Accuracy: {acc:.4f}\n')
            
            # Add confusion matrix
            log_file.write('\nConfusion Matrix:\n')
            log_file.write('Predicted ?\n')
            log_file.write('True ?    ' + ' '.join([f'{cls:8s}' for cls in class_names]) + '\n')
            
            for i in range(num_classes):
                row = [f'{class_names[i]:8s}']
                for j in range(num_classes):
                    row.append(f'{confusion_matrix[i][j]:8.0f}')
                log_file.write(' '.join(row) + '\n')
            log_file.write('\n')
    
    if print_results:
        print(f'Overall Accuracy: {epoch_acc:.4f}')
        for cls_idx, cls_name in enumerate(class_names):
            if class_total[cls_idx] > 0:
                acc = class_correct[cls_idx] / class_total[cls_idx]
            else:
                acc = 0.0
            print(f'Class {cls_name} - Correct: {class_correct[cls_idx]}, Total: {class_total[cls_idx]}, Accuracy: {acc:.4f}')
        
        # Print confusion matrix
        print('\nConfusion Matrix:')
        print('Predicted ?')
        print('True ?    ' + ' '.join([f'{cls:8s}' for cls in class_names]))
        for i in range(num_classes):
            row = [f'{class_names[i]:8s}']
            for j in range(num_classes):
                row.append(f'{confusion_matrix[i][j]:8.0f}')
            print(' '.join(row))
    
    return epoch_loss, epoch_acc, class_correct, class_total


# Training loop
def train_model(model, train_loader, val_loader, test_loader, criterion, optimizer, scheduler, num_epochs, device, log_file_path):
    best_val_acc = 0.0
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }

    with open(log_file_path, 'a') as log_file:
        log_file.write(f"Starting training for {num_epochs} epochs\n")
        print(f"Starting training for {num_epochs} epochs")
        
        for epoch in range(num_epochs):
            print(f"Training for epoch: {epoch}")
            # Train
            train_loss, train_acc = train_one_epoch(
                model, train_loader, criterion, optimizer, device
            )

            # Validate
            val_loss, val_acc, _, _ = validate(
                model, val_loader, criterion, device, epoch,log_file_path
            )

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
                torch.save(model.state_dict(), 'best_model_128_4_WF_loss.pth')

            # Print and log epoch results
            epoch_results = (
                f'Epoch {epoch+1}/{num_epochs} | '
                f'Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f} | '
                f'Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}'
            )
            print(epoch_results)
            log_file.write(f"{epoch_results}\n")

    # Load best model for final evaluation
    model.load_state_dict(torch.load('best_model_128_4_WF_loss.pth'))
    
    # Final evaluation on test set
    print("\nEvaluating on test set:")
    with open(log_file_path, 'a') as log_file:
        log_file.write("\nTest Set Evaluation:\n")
    
    test_loss, test_acc, class_correct, class_total = validate(
        model, test_loader, criterion, device, epoch,log_file_path, print_results=True
    )
    
    print(f"\nBest validation accuracy: {best_val_acc:.4f}")
    print(f"Final test accuracy: {test_acc:.4f}")
    
    with open(log_file_path, 'a') as log_file:
        log_file.write(f"\nBest validation accuracy: {best_val_acc:.4f}\n")
        log_file.write(f"Final test accuracy: {test_acc:.4f}\n")
    
    return history, test_acc, class_correct, class_total


# Run the training
history = train_model(
    model, train_dataloader, val_dataloader,
    test_dataloader,
    criterion, optimizer, scheduler,
    num_epochs, device, "vit_vanilla_128_4_WF_loss.txt"
)


