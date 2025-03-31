"""
This script is used to construct the NN classifier.
The input embeddings are passed through a CrossAttentionLayer to reduce them to a single vector of dim=512.
There are 4 different models to choose from: FCNN, ResidualFCNN, AttentionFCNN, TransformerFCNN.
They all have Cross-Attention implemented in the first layer, to make them agnostic to the order of input embeddings.

Example instantiations:
embedding_dim = 512 (BWSE length)
output_dim = 20 (how many conflict groups we have)

fcnn = FCNN(embedding_dim, output_dim)
residual_fcnn = ResidualFCNN(embedding_dim, output_dim)
attention_fcnn = AttentionFCNN(embedding_dim, output_dim)
transformer_fcnn = TransformerFCNN(embedding_dim, output_dim)

Example usage:
bazel run -c opt experimental/childebrandt/ayildiz/classifier/model1 -- 
"""

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from torch.utils.data import DataLoader, Dataset, random_split
from torchmetrics.classification import MultilabelPrecision, MultilabelRecall
import pandas as pd
import numpy as np
from tqdm import tqdm

import os
from pathlib import Path
from datetime import datetime

from torch.utils.tensorboard import SummaryWriter

# Set device to GPU if available
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f"#### Using device: {device}")

# Custom Dataset
class BlobDataset(Dataset):
    def __init__(self, X_df, Y_df, max_seq_len=None):
        self.blob_ids = X_df['blob_id'].unique()  # Unique blob_ids
        self.labels = {row['blob_id']: row[1:].values.astype(float) for _, row in Y_df.iterrows()}
        self.model_embedding_dim = X_df['embedding'].iloc[0].shape[0]
        self.model_output_dim = Y_df.shape[1] - 1
        
        # Determine the maximum sequence length if not provided
        if max_seq_len is None:
            max_seq_len = max(X_df.groupby('blob_id').size())
        self.max_seq_len = max_seq_len

        # Preprocess and cache embeddings with padding and attention masks
        self.embeddings_cache = []
        self.masks_cache = []
        for blob_id in tqdm(self.blob_ids, desc="Caching embeddings"):
            embeddings = np.array(X_df[X_df['blob_id'] == blob_id]['embedding'].tolist())
            embeddings = torch.tensor(embeddings, dtype=torch.float32)
            
            # Padding the sequence to the maximum length
            padded_embeddings = torch.zeros((self.max_seq_len, self.model_embedding_dim))
            mask = torch.zeros(self.max_seq_len, dtype=torch.bool)
            seq_len = min(embeddings.size(0), self.max_seq_len)
            padded_embeddings[:seq_len] = embeddings[:seq_len]
            mask[:seq_len] = 1

            self.embeddings_cache.append(padded_embeddings)
            self.masks_cache.append(mask)

    def __len__(self):
        return len(self.blob_ids)

    def __getitem__(self, idx):
        # Retrieve preprocessed padded embeddings and corresponding mask and label
        padded_embedding = self.embeddings_cache[idx]
        mask = self.masks_cache[idx]
        blob_id = self.blob_ids[idx]
        label = torch.tensor(self.labels[blob_id], dtype=torch.float32)
        return padded_embedding, mask, label


class CrossAttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(CrossAttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=4, batch_first=True)

    def forward(self, x, mask):
        # Mask must be inverted for PyTorch MultiheadAttention (True = ignore, False = attend)
        attn_output, attn_output_weights = self.attention(x, x, x, key_padding_mask=~mask)

        # Mean pooling over attended outputs
        # mean_pool = torch.mean(attn_output, dim=1)
        # return mean_pool

        # Weighted average of input
        # weighted_embedding = torch.bmm(attn_output_weights, x).sum(dim=1)  # Shape: (batch_size, input_dim)
        # return weighted_embedding
       
        # Weighted average of attended outputs
        weighted_attn_output = torch.bmm(attn_output_weights, attn_output).sum(dim=1)
        return weighted_attn_output

class FCNN(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(FCNN, self).__init__()
        print("### Using FCNN")
        self.cross_attention = CrossAttentionLayer(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 512)
        self.bn1 = nn.BatchNorm1d(512)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.fc3 = nn.Linear(256, 128)
        self.bn3 = nn.BatchNorm1d(128)
        self.fc4 = nn.Linear(128, 64)
        self.bn4 = nn.BatchNorm1d(64)
        self.out = nn.Linear(64, output_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x, mask):
        x = self.cross_attention(x, mask)  # Apply cross-attention and mean pooling
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = self.dropout(F.relu(self.bn3(self.fc3(x))))
        x = self.dropout(F.relu(self.bn4(self.fc4(x))))
        x = torch.sigmoid(self.out(x))
        return x

class ResidualBlock(nn.Module):
    def __init__(self, in_dim):
        super(ResidualBlock, self).__init__()
        self.fc1 = nn.Linear(in_dim, in_dim)
        self.bn1 = nn.BatchNorm1d(in_dim)
        self.fc2 = nn.Linear(in_dim, in_dim)
        self.bn2 = nn.BatchNorm1d(in_dim)
        self.dropout = nn.Dropout(0.3)

    def forward(self, x):
        identity = x
        out = self.dropout(F.relu(self.bn1(self.fc1(x))))
        out = self.bn2(self.fc2(out))
        out += identity
        return F.relu(out)

class ResidualFCNN(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(ResidualFCNN, self).__init__()
        print("### Using Residual FCNN")
        self.cross_attention = CrossAttentionLayer(embedding_dim)
        self.res_block1 = ResidualBlock(embedding_dim)
        self.res_block2 = ResidualBlock(embedding_dim)
        self.fc = nn.Linear(embedding_dim, 256)
        self.bn = nn.BatchNorm1d(256)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(256, output_dim)

    def forward(self, x, mask):
        x = self.cross_attention(x, mask)  # Apply cross-attention and mean pooling
        x = self.res_block1(x)
        x = self.dropout(x)
        x = self.res_block2(x)
        x = self.dropout(x)
        x = self.dropout(F.relu(self.bn(self.fc(x))))
        x = torch.sigmoid(self.out(x))
        return x

class AttentionLayer(nn.Module):
    def __init__(self, input_dim):
        super(AttentionLayer, self).__init__()
        self.attention = nn.MultiheadAttention(input_dim, num_heads=1, batch_first=True)

    def forward(self, x):
        x = x.unsqueeze(1)  # Adding sequence dimension
        attn_output, _ = self.attention(x, x, x)
        return attn_output.squeeze(1)  # Removing sequence dimension

class AttentionFCNN(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(AttentionFCNN, self).__init__()
        print("### Using Attention FCNN")
        self.cross_attention = CrossAttentionLayer(embedding_dim)
        self.attention = AttentionLayer(embedding_dim)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x, mask):
        x = self.cross_attention(x, mask)  # Apply cross-attention and mean pooling
        x = self.attention(x)
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = torch.sigmoid(self.out(x))
        return x

class TransformerFCNN(nn.Module):
    def __init__(self, embedding_dim, output_dim):
        super(TransformerFCNN, self).__init__()
        print("### Using Transformer FCNN")
        self.cross_attention = CrossAttentionLayer(embedding_dim)
        self.transformer_layer = nn.TransformerEncoderLayer(d_model=embedding_dim, nhead=4, batch_first=True)
        self.fc1 = nn.Linear(embedding_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.fc2 = nn.Linear(256, 128)
        self.bn2 = nn.BatchNorm1d(128)
        self.dropout = nn.Dropout(0.3)
        self.out = nn.Linear(128, output_dim)

    def forward(self, x, mask):
        x = self.cross_attention(x, mask)  # Apply cross-attention and mean pooling
        x = x.unsqueeze(1)  # Adding sequence dimension for transformer
        x = self.transformer_layer(x).squeeze(1)  # Passing through transformer and removing sequence dimension
        x = self.dropout(F.relu(self.bn1(self.fc1(x))))
        x = self.dropout(F.relu(self.bn2(self.fc2(x))))
        x = torch.sigmoid(self.out(x))
        return x

# Function to split the DataLoader into training and validation loaders
def split_dataloader(dataset, split_ratio=0.7, batch_size=1, shuffle=True):
    train_size = int(split_ratio * len(dataset))
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=shuffle)
    val_loader = DataLoader(val_dataset, batch_size=batch_size, shuffle=shuffle)

    return train_loader, val_loader

# Function to train a single model
def train_model(model, model_arch, model_full_name, train_loader, val_loader, epochs, lr, weight_decay):
    # Initialize TensorBoard writer for each model
    log_dir = f"{Path.home()}/tb/runs/{model_full_name}_exprmnt_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
    writer = SummaryWriter(log_dir=log_dir)
    
    criterion = nn.BCELoss()
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    # scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)  # Learning rate scheduler
    
    best_val_loss = float('inf')  # Track the best validation loss
    # Assuming the number of output classes is output_dim
    precision_metric = MultilabelPrecision(num_labels=model.out.out_features, average='macro').to(device)
    recall_metric = MultilabelRecall(num_labels=model.out.out_features, average='macro').to(device)

    model.to(device)  # Move model to GPU
    
    for epoch in tqdm(range(epochs), desc=f"Training epochs - {model_full_name}"):
        model.train()
        total_train_loss = 0.0
        # Reset metrics at the start of the epoch
        precision_metric.reset()
        recall_metric.reset()

        # Training phase
        for i, (embeddings, masks, labels) in enumerate(train_loader):
            embeddings = embeddings.to(device)
            masks = masks.to(device)
            labels = labels.to(device)

            optimizer.zero_grad()
            outputs = model(embeddings, masks)
            loss = criterion(outputs, labels)

            loss.backward()
            # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)  # Apply gradient clipping to prevent exploding gradients
            optimizer.step()
            total_train_loss += loss.item()

            # Update metrics
            precision_metric.update(outputs, labels.int())
            recall_metric.update(outputs, labels.int())

        # Compute precision and recall for the entire training set
        train_precision = precision_metric.compute().item()
        train_recall = recall_metric.compute().item()

        # Log precision and recall to TensorBoard
        writer.add_scalar(f"{model_arch}/lr={lr}/Train Precision", train_precision, epoch)
        writer.add_scalar(f"{model_arch}/lr={lr}/Train Recall", train_recall, epoch)
            
        # Validation phase
        model.eval()
        total_val_loss = 0.0
        # Reset metrics at the start of the epoch
        precision_metric.reset()
        recall_metric.reset()
        with torch.no_grad():  # Disable gradient calculation for validation
            for embeddings, masks, labels in val_loader:
                embeddings = embeddings.to(device)
                masks = masks.to(device)
                labels = labels.to(device)

                outputs = model(embeddings, masks)
                loss = criterion(outputs, labels)
                total_val_loss += loss.item()

                # Update metrics
                precision_metric.update(outputs, labels.int())
                recall_metric.update(outputs, labels.int())

        avg_train_loss = total_train_loss / len(train_loader)
        avg_val_loss = total_val_loss / len(val_loader)

        # # Step the learning rate scheduler
        # scheduler.step(avg_val_loss)

        # Compute precision and recall for the entire training set
        val_precision = precision_metric.compute().item()
        val_recall = recall_metric.compute().item()

        # Log average losses to TensorBoard
        writer.add_scalar(f"{model_arch}/lr={lr}/Avg Train Loss", avg_train_loss, epoch)
        writer.add_scalar(f"{model_arch}/lr={lr}/Avg Val Loss", avg_val_loss, epoch)

        # Log precision and recall to TensorBoard
        writer.add_scalar(f"{model_arch}/lr={lr}/Val Precision", val_precision, epoch)
        writer.add_scalar(f"{model_arch}/lr={lr}/Val Recall", val_recall, epoch)

        # Save model at every Nth epoch with a timestamp
        if (epoch + 1) % 50 == 0:
            print(f"Epoch {epoch+1}/{epochs}\n Train Loss: {avg_train_loss}\n Val Loss: {avg_val_loss}")

            timestamp = datetime.now().strftime('%Y%m%d_%H%M%S')
            model_save_path = f"{log_dir}/model_epoch_{epoch+1}_date_{timestamp}_valLoss_{avg_val_loss}.pth"
            torch.save(model.state_dict(), model_save_path)
            print(f"Model saved at {model_save_path}")

    # Close the TensorBoard writer
    writer.close()
    return model


if __name__ == "__main__":
    # Example embeddings
    X_df = pd.DataFrame({
        'blob_id': ['a', 'a', 'b', 'c', 'c', 'c', 'd', 'e', 'f'],
        'embedding': [np.random.rand(512) for _ in range(9)]
    })

    # Example labels
    Y_df = pd.DataFrame({
        'blob_id': ['a', 'b', 'c', 'd', 'e', 'f'],
        'label1': [1, 0, 1, 0, 1, 0],
        'label2': [0, 1, 1, 0, 1, 0],
        'label3': [1, 0, 0, 1, 0, 1],
        'label4': [0, 1, 1, 0, 1, 0],
        'label5': [1, 0, 0, 1, 0, 1],
        'label6': [0, 1, 1, 0, 1, 0],
        'label7': [1, 0, 1, 0, 1, 0],
        'label8': [0, 1, 0, 1, 0, 1],
        'label9': [1, 0, 0, 1, 0, 1],
        'label10': [0, 1, 1, 0, 1, 0],
        'label11': [1, 0, 0, 1, 0, 1],
    })

    # Instantiate Dataset
    dataset = BlobDataset(X_df, Y_df)

    # Split the dataset and get DataLoaders directly
    train_loader, val_loader = split_dataloader(dataset, split_ratio=0.67, batch_size=2, shuffle=True)

    # Initialize and train the model (will likely overfit on this dummy data)
    model = FCNN(embedding_dim=dataset.model_embedding_dim, output_dim=dataset.model_output_dim)
    model = train_model(model, "dummy training", "sample model", train_loader, val_loader, epochs=1000, lr=0.0001, weight_decay=0.01)

    import ipdb; ipdb.set_trace()