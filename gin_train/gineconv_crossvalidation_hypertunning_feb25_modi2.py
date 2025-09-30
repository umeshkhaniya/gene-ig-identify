import optuna
import torch
import numpy as np
import json
import random
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.nn import Linear, BatchNorm1d, Dropout, ReLU, Sequential
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool
from sklearn.model_selection import train_test_split, StratifiedShuffleSplit
from sklearn.metrics import classification_report
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
import torch_geometric.data.data

#torch.serialization.add_safe_globals([torch_geometric.data.data.DataEdgeAttr])
edge_in_channels_featutes = 8 
node_features  = 1320

num_igtpyes = 12 # here I exclude ORF but include jellyroll

# Set seed
def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

set_seed(42)

# Load graphs
# all_graphs = torch.load("../all_graphs_contact.pt")
# graph_lookup = torch.load("../graph_lookup_contact.pt")

# all_graphs = torch.load("../all_graphs.pt")
# graph_lookup = torch.load("../graph_lookup.pt")

all_graphs = torch.load("../all_graphs_nocor.pt", weights_only=False)
graph_lookup = torch.load("../graph_lookup_nocor.pt", weights_only=False)

# all_graphs = torch.load("../all_graphs_nocor_cbeta.pt")
# graph_lookup = torch.load("../graph_lookup_nocor_cbeta.pt")
graph_list = list(graph_lookup.values())

# Extract labels for stratification
labels = [data.y.item() for data in graph_list]





# Step 1: Train-Val-Test Split (80% Train+Val, 20% Test)
train_val_idx, test_idx = train_test_split(
    range(len(graph_list)), test_size=0.2, stratify=labels, random_state=42
)

# Ensure test set is separated
train_val_graphs = [graph_list[i] for i in train_val_idx]
test_graphs = [graph_list[i] for i in test_idx]

# Save test data and labels
# test_labels = [labels[i] for i in test_idx]
# torch.save(test_graphs, "test_graphs.pt")
# torch.save(test_labels, "test_labels.pt")
# Save test data and labels
test_labels = [labels[i] for i in test_idx]

torch.save(test_graphs, "test_graphs.pt")
torch.save(test_labels, "test_labels.pt")
# torch.save(test_graphs, "test_graphs_contact.pt")
# torch.save(test_labels, "test_labels_contact.pt")


# Extract labels for stratification for train+val set
train_val_labels = [labels[i] for i in train_val_idx]

# Step 2: Split Train-Val (e.g., 80% Train, 20% Val) from the train+val set
sss = StratifiedShuffleSplit(n_splits=1, test_size=0.2, random_state=42)
train_idx, val_idx = next(sss.split(train_val_graphs, train_val_labels))

# Now create train and validation sets
train_graphs = [train_val_graphs[i] for i in train_idx]
val_graphs = [train_val_graphs[i] for i in val_idx]


# calculate the loss function for train graph.

# Compute class weights for imbalanced dataset
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class_counts = torch.bincount(torch.tensor([data.y.item() for data in train_graphs]))
class_weights = 1.0 / torch.log1p(class_counts.float())
loss_weight = class_weights / class_weights.sum()
loss_weight = loss_weight.to(device)
print(f'Class weights: {loss_weight}')


def check_data_validity(graph_list):
    for i, data in enumerate(graph_list):
        if torch.isnan(data.x).any() or torch.isinf(data.x).any():
            print(f"NaN or Inf found in node features of graph {i}")
        if torch.isnan(data.edge_attr).any() or torch.isinf(data.edge_attr).any():
            print(f"NaN or Inf found in edge attributes of graph {i}")
        if torch.isnan(data.y).any() or torch.isinf(data.y).any():
            print(f"NaN or Inf found in labels of graph {i}")
check_data_validity(all_graphs)
#check_data_validity(val_graphs)



# Print dataset sizes
print(f"Training set: {len(train_graphs)} graphs")
print(f"Validation set: {len(val_graphs)} graphs")
print(f"Test set: {len(test_graphs)} graphs")


class GraphClassifier(torch.nn.Module):
    def __init__(self, in_channels, edge_in_channels, hidden_dim, num_classes, num_layers, dropout_rate):
        super(GraphClassifier, self).__init__()

        self.convs = torch.nn.ModuleList()

        # First GINEConv layer
        self.convs.append(GINEConv(
            Sequential(
                Linear(in_channels, hidden_dim),
                BatchNorm1d(hidden_dim),
                ReLU(),
                Dropout(dropout_rate),
                Linear(hidden_dim, hidden_dim),
            ), edge_dim=edge_in_channels
        ))

        # Additional GINEConv layers
        for _ in range(1, num_layers):
            self.convs.append(GINEConv(
                Sequential(
                    Linear(hidden_dim, hidden_dim),
                    BatchNorm1d(hidden_dim),
                    ReLU(),
                    Dropout(dropout_rate),
                    Linear(hidden_dim, hidden_dim),
                ), edge_dim=edge_in_channels
            ))

        # Fully connected layers after concatenated hybrid pooling
        pool_output_dim = hidden_dim * 3 * num_layers  # mean + add + max for each layer
        #pool_output_dim = hidden_dim * 2 * num_layers  # mean +  max for each layer
        #pool_output_dim = hidden_dim *  num_layers
        self.lin1 = Linear(pool_output_dim, pool_output_dim)
        self.lin2 = Linear(pool_output_dim, num_classes)

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch

        pooled_outputs = []
        for conv in self.convs:
            x = conv(x, edge_index, edge_attr)
            x = F.relu(x)

            # Pool using mean, add, and max for each layer
            pooled_outputs.append(global_mean_pool(x, batch))
            pooled_outputs.append(global_add_pool(x, batch))
            pooled_outputs.append(global_max_pool(x, batch))

        # Concatenate pooled outputs
        h = torch.cat(pooled_outputs, dim=1)

        # Classification head
        h = self.lin1(h)
        h = F.relu(h)
        h = F.dropout(h, p=0.5, training=self.training)
        h = self.lin2(h)

        return h


def objective(trial):
    """Hyperparameter optimization function."""

    # Sample hyperparameters
    hidden_dim = trial.suggest_categorical("hidden_dim", [64, 128, 256])
    dropout_rate = trial.suggest_float("dropout", 0.1, 0.5)
    lr = trial.suggest_float("lr", 1e-4, 1e-2, log=True)
    weight_decay = trial.suggest_float("weight_decay", 1e-5, 1e-3, log=True)
    batch_size = trial.suggest_categorical("batch_size", [32, 64, 128])
    num_layers = trial.suggest_int("num_layers", 2, 4)

    # Create DataLoader
    train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
    val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

    # Model
    model = GraphClassifier(node_features, edge_in_channels_featutes, hidden_dim,
                            num_igtpyes, num_layers, dropout_rate).to(device)
    optimizer = optim.AdamW(model.parameters(), lr=lr, weight_decay=weight_decay)
    scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
    loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight)

    # Training loop
    for epoch in range(20):  # Short training for tuning
        model.train()
        for data in train_loader:
            data = data.to(device)
            optimizer.zero_grad()
            out = model(data)
            loss = loss_fn(out, data.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            optimizer.step()
        scheduler.step(loss)

    # Validation
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            loss = loss_fn(out, data.y)
            val_loss += loss.item()

            preds = out.argmax(dim=1)
            correct += (preds == data.y).sum().item()
            total += data.y.size(0)

    val_loss /= len(val_loader)
    val_accuracy = correct / total

    # You can choose to either:
    # - minimize val_loss (for stability)
    # - or maximize val_accuracy (add `direction="maximize"` in create_study)
    return val_loss  # Or: return -val_accuracy

# Run Optuna
study = optuna.create_study(direction="minimize")
study.optimize(objective, n_trials=30)

best_params = study.best_params


# Train final model with best hyperparameters
batch_size = best_params["batch_size"]
train_loader = DataLoader(train_graphs, batch_size=batch_size, shuffle=True)
val_loader = DataLoader(val_graphs, batch_size=batch_size, shuffle=False)

# Initialize model with best hyperparameters
model = GraphClassifier(
    in_channels=node_features, 
    edge_in_channels=edge_in_channels_featutes, 
    hidden_dim=best_params["hidden_dim"], 
    num_classes=num_igtpyes, 
    num_layers=best_params["num_layers"], 
    dropout_rate=best_params["dropout"]
).to(device)

optimizer = optim.AdamW(model.parameters(), lr=best_params["lr"], weight_decay=best_params["weight_decay"])
scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=5)
loss_fn = torch.nn.CrossEntropyLoss(weight=loss_weight)

# Tracking variables for plotting
train_losses = []
val_accuracies = []

# Full training loop
num_epochs = 100
for epoch in range(num_epochs):
    model.train()
    total_loss = 0
    for data in train_loader:
        data = data.to(device)
        optimizer.zero_grad()
        out = model(data)
        loss = loss_fn(out, data.y)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()
        total_loss += loss.item()
    
    scheduler.step(total_loss)
    current_lr = optimizer.param_groups[0]['lr']

    # Validation accuracy
    model.eval()
    correct = 0
    total = 0
    with torch.no_grad():
        for data in val_loader:
            data = data.to(device)
            out = model(data)
            preds = out.argmax(dim=1)
            correct += (preds == data.y).sum().item()
            total += data.y.size(0)

    val_acc = correct / total

    # Logging
    print(f"Epoch {epoch+1}, Loss: {total_loss:.4f}, Val Accuracy: {val_acc:.4f}, Learning Rate: {current_lr:.6f}")

    # Store for plotting
    train_losses.append(total_loss)
    val_accuracies.append(val_acc)

# Save the final model and hyperparameters
torch.save(model.state_dict(), "best_graph_model.pth")
print("Final trained model saved!")

# Save best hyperparameters and model config
model_config = {
    "best_params": best_params,
    "node_features": node_features,
    "edge_in_channels_featutes": edge_in_channels_featutes,
    "num_classes": num_igtpyes
}
with open("model_config.json", "w") as f:
    json.dump(model_config, f)
with open("best_hyperparameters.json", "w") as f:
    json.dump(best_params, f)

# Plotting
fig, ax1 = plt.subplots(figsize=(10, 6))
ax1.plot(range(1, num_epochs + 1), train_losses, label='Training Loss', marker='o', color='blue')
ax1.set_xlabel('Epoch')
ax1.set_ylabel('Training Loss', color='blue')
ax1.tick_params(axis='y', labelcolor='blue')

ax2 = ax1.twinx()
ax2.plot(range(1, num_epochs + 1), val_accuracies, label='Validation Accuracy', marker='o', color='orange')
ax2.set_ylabel('Validation Accuracy', color='orange')
ax2.tick_params(axis='y', labelcolor='orange')

fig.suptitle('Training Loss and Validation Accuracy per Epoch')
fig.legend(loc='upper center', bbox_to_anchor=(0.5, -0.05), ncol=2)
plt.grid(True)
plt.savefig('loss_accuracy_plot_hybrid.png')
#plt.show()











