# ================================================================
# Multi-output Brain Tumor Model: PyTorch (Improved Version)
# ================================================================

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler, LabelEncoder
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, TensorDataset

# -------------------------------
# Step 1: Load Data
# -------------------------------
df = pd.read_csv("Brain_Tumor_Prediction_Dataset.csv")  # replace path

# -------------------------------
# Step 2: Select Features & Targets
# -------------------------------
features = ['Age', 'Tumor_Size', 'Tumor_Growth_Rate', 'Symptom_Severity', 
            'Tumor_Location', 'MRI_Findings', 'Radiation_Exposure']

target_survival = 'Survival_Rate(%)'   # regression
target_treatment = 'Treatment_Received' # classification

X = df[features].copy()
y_surv = df[target_survival].values / 100.0  # normalize to 0-1
y_trt = df[target_treatment].values

# -------------------------------
# Step 3: Preprocess Features
# -------------------------------

# Map ordinal features
growth_map = {'Slow': 0, 'Moderate': 1, 'Rapid': 2}
severity_map = {'Mild': 0, 'Moderate': 1, 'Severe': 2}

X['Tumor_Growth_Rate'] = X['Tumor_Growth_Rate'].map(growth_map)
X['Symptom_Severity'] = X['Symptom_Severity'].map(severity_map)

# Numeric features
numeric_features = ['Age', 'Tumor_Size', 'Tumor_Growth_Rate', 'Symptom_Severity']
scaler = StandardScaler()
X_numeric = scaler.fit_transform(X[numeric_features])

# Categorical features as label-encoded for embedding
categorical_features = ['Tumor_Location', 'MRI_Findings', 'Radiation_Exposure']
cat_encoders = {}
X_cat_indices = []

for col in categorical_features:
    le = LabelEncoder()
    X[col] = X[col].astype(str)
    X_cat_indices.append(le.fit_transform(X[col]))
    cat_encoders[col] = le

X_cat_indices = np.stack(X_cat_indices, axis=1)

# -------------------------------
# Step 4: Encode treatment target
# -------------------------------
le_trt = LabelEncoder()
y_trt_indices = le_trt.fit_transform(y_trt)

# -------------------------------
# Step 5: Convert to PyTorch tensors
# -------------------------------
X_numeric_tensor = torch.tensor(X_numeric, dtype=torch.float32)
X_cat_tensor = torch.tensor(X_cat_indices, dtype=torch.long)
y_surv_tensor = torch.tensor(y_surv, dtype=torch.float32).unsqueeze(1)
y_trt_tensor = torch.tensor(y_trt_indices, dtype=torch.long)

# -------------------------------
# Step 6: Train/Test Split
# -------------------------------
X_num_train, X_num_test, X_cat_train, X_cat_test, y_surv_train, y_surv_test, y_trt_train, y_trt_test = train_test_split(
    X_numeric_tensor, X_cat_tensor, y_surv_tensor, y_trt_tensor, test_size=0.2, random_state=42
)

train_dataset = TensorDataset(X_num_train, X_cat_train, y_surv_train, y_trt_train)
test_dataset = TensorDataset(X_num_test, X_cat_test, y_surv_test, y_trt_test)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)

# -------------------------------
# Step 7: Build Multi-output Model with Embeddings
# -------------------------------
class MultiOutputNN(nn.Module):
    def __init__(self, num_numeric, cat_dims, embedding_dims, hidden_dim=128):
        super(MultiOutputNN, self).__init__()
        # Embeddings for categorical features
        self.embeddings = nn.ModuleList([nn.Embedding(cat_dims[i], embedding_dims[i]) for i in range(len(cat_dims))])
        emb_total_dim = sum(embedding_dims)
        
        # Shared layers
        self.shared = nn.Sequential(
            nn.Linear(num_numeric + emb_total_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(0.2),
            nn.Linear(hidden_dim, hidden_dim // 2),
            nn.ReLU(),
            nn.Dropout(0.2)
        )
        # Heads
        self.survival_head = nn.Linear(hidden_dim // 2, 1)
        self.treatment_head = nn.Linear(hidden_dim // 2, len(le_trt.classes_))
    
    def forward(self, x_num, x_cat):
        x_emb = [emb(x_cat[:,i]) for i, emb in enumerate(self.embeddings)]
        x_emb = torch.cat(x_emb, dim=1)
        x = torch.cat([x_num, x_emb], dim=1)
        x = self.shared(x)
        surv = self.survival_head(x)
        trt = self.treatment_head(x)
        return surv, trt

# Embedding sizes
cat_dims = [len(le.classes_) for le in cat_encoders.values()]
embedding_dims = [min(50, (dim+1)//2) for dim in cat_dims]

model = MultiOutputNN(len(numeric_features), cat_dims, embedding_dims)
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
model.to(device)

# -------------------------------
# Step 8: Losses and Optimizer
# -------------------------------
criterion_surv = nn.MSELoss()
criterion_trt = nn.CrossEntropyLoss()
optimizer = optim.Adam(model.parameters(), lr=0.001)

# -------------------------------
# Step 9: Training Loop
# -------------------------------
epochs = 50
for epoch in range(epochs):
    model.train()
    running_loss = 0
    for X_num_batch, X_cat_batch, y_surv_batch, y_trt_batch in train_loader:
        X_num_batch = X_num_batch.to(device)
        X_cat_batch = X_cat_batch.to(device)
        y_surv_batch = y_surv_batch.to(device)
        y_trt_batch = y_trt_batch.to(device)
        
        optimizer.zero_grad()
        y_surv_pred, y_trt_pred = model(X_num_batch, X_cat_batch)
        loss_surv = criterion_surv(y_surv_pred, y_surv_batch)
        loss_trt = criterion_trt(y_trt_pred, y_trt_batch)
        loss = loss_surv + loss_trt
        loss.backward()
        optimizer.step()
        running_loss += loss.item()
    
    print(f"Epoch {epoch+1}/{epochs}, Loss: {running_loss/len(train_loader):.4f}")

torch.save(model.state_dict(), "multi_output_brain_tumor_model.pth")
# -------------------------------
# Step 10: Evaluation
# -------------------------------
model.eval()
surv_preds, trt_preds, surv_true, trt_true = [], [], [], []

with torch.no_grad():
    for X_num_batch, X_cat_batch, y_surv_batch, y_trt_batch in test_loader:
        X_num_batch = X_num_batch.to(device)
        X_cat_batch = X_cat_batch.to(device)
        y_surv_batch = y_surv_batch.to(device)
        y_trt_batch = y_trt_batch.to(device)
        
        y_surv_pred, y_trt_pred = model(X_num_batch, X_cat_batch)
        surv_preds.extend(y_surv_pred.cpu().numpy())
        trt_preds.extend(torch.argmax(y_trt_pred, dim=1).cpu().numpy())
        surv_true.extend(y_surv_batch.cpu().numpy())
        trt_true.extend(y_trt_batch.cpu().numpy())

from sklearn.metrics import mean_absolute_error, accuracy_score
surv_mae = mean_absolute_error(np.array(surv_true)*100, np.array(surv_preds)*100)
trt_acc = accuracy_score(trt_true, trt_preds)
print(f"Test Survival MAE: {surv_mae:.2f}%")
print(f"Test Treatment Accuracy: {trt_acc:.2f}")

# -------------------------------
# Step 11: Example Predictions
# -------------------------------
for i in range(5):
    pred_surv = surv_preds[i][0]*100
    pred_trt = le_trt.inverse_transform([trt_preds[i]])[0]
    print(f"Predicted Survival: {pred_surv:.2f}%, Predicted Treatment: {pred_trt}")
