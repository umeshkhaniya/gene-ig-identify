import torch
import json
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.nn import Linear, BatchNorm1d, Dropout, ReLU, Sequential
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay
#from torch.serialization import add_safe_globals
#from torch_geometric.data import Data
##add_safe_globals([Data])

edge_in_channels_featutes = 8 
node_features  = 1320 # last three coordinates

num_igtpyes = 12 # here I exclude IgFN3-like
test_graphs = torch.load("test_graphs.pt",weights_only=False)
test_labels = torch.load("test_labels.pt", weights_only=False)

# # Load test dataset and labels
# test_graphs = torch.load("test_graphs_contact.pt")
# test_labels = torch.load("test_labels_contact.pt")

#check covid_dataset: 
# test_graphs = torch.load("../all_graphs_antobodies_covid_diverse.pt")
# test_labels = torch.load("../graph_lookup_antobodies_covid_diverse.pt")

# Print dataset sizes

print(f"Test set: {len(test_graphs)} graphs")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best hyperparameters
with open("best_hyperparameters.json", "r") as f:
    best_params = json.load(f)

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


# Initialize model with best hyperparameters
model = GraphClassifier(
    in_channels=node_features, 
    edge_in_channels=edge_in_channels_featutes, 
    hidden_dim=best_params["hidden_dim"], 
    num_classes=num_igtpyes, 
    num_layers=best_params["num_layers"], 
    dropout_rate=best_params["dropout"]
).to(device)


# Load trained model weights
model.load_state_dict(torch.load("best_graph_model.pth", map_location=device))
model.eval()

test_loader = DataLoader(test_graphs, batch_size=best_params["batch_size"], shuffle=False)


graph_names = [data.unique_name_file for data in test_graphs]

# Evaluate on test set
all_preds = [] # prediction label
all_probs = [] # all predicted class probability
all_labels = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        probs = F.softmax(out, dim=1)
        preds = probs.argmax(dim=1)

        all_probs.extend(probs.cpu().numpy())
        all_preds.extend(preds.cpu().numpy())
        all_labels.extend(data.y.cpu().numpy())



#Convert to numpy arrays for easy indexing
all_labels = np.array(all_labels)
all_preds = np.array(all_preds)
all_probs = np.array(all_probs)




# Extract graph names from the dataset
graph_names = [data.unique_name_file for data in test_graphs]

# # Define the label mapping
# label_mapping = {'IgV':0, 'IgC1': 1, 'Cadherin': 2, 'IgE':3,  'IgFN3-like':4, 
#     'SOD':5, 'IgI':6, 'ORF':7, 'IgFN3':8, 'IgC2':9, 'CD19':10, 'Lamin':11}

# label_mapping = {'IgV':0, 'IgC1': 1, 'Cadherin': 2, 'IgE':3, 
#     'SOD':4, 'IgI':5, 'ORF':6, 'IgFN3':7, 'IgC2':8, 'CD19':9, 'Lamin':10}

#label_mapping = {'IgV': 0, 'IgC1': 1, 'IgC2': 2, 'IgI': 3, 'IgE': 4, 'Cadherin': 5, 'IgFN3': 6,
#                 'Lamin': 7, 'SOD': 8}

label_mapping = {'IgV':0, 'IgC1': 1,  'IgC2':2,'IgI':3, 'IgE':4,'Cadherin': 5, 'IgFN3':6,
    'Lamin':7,  'SOD':8,'IgFN3-like':9, 'CD19':10, "JellyRoll":11} 
# Reverse mapping from number to name
reverse_label_mapping = {v: k for k, v in label_mapping.items()}


# # Find misclassified indices where true labels and predicted label doesn't match
misclassified_idx = np.where(all_labels != all_preds)[0]


print("\n🔴 Misclassified Samples:")
print(f"Graph True Label  Predicted Label Predicted probability")
for idx in misclassified_idx:
    true_label_name = reverse_label_mapping[all_labels[idx]]
    predicted_label_name = reverse_label_mapping[all_preds[idx]]
    probability= f"{max(all_probs[idx]):.3f}"

    print(f"{graph_names[idx]} {true_label_name} {predicted_label_name} {probability}")
    for class_idx, prob in enumerate(all_probs[idx]):
        if prob > 0.1:
            class_name = reverse_label_mapping[class_idx]
            print(f"{class_name}: {prob:.3f}")
    print("____________________")


# Find correctly classified indices
correct_idx = np.where(all_labels == all_preds)[0]

print("\n✅ Correctly Classified 10 Samples graphs :")
for idx in correct_idx[:10]:  # Print first 10 correct samples
    correct_label_name = reverse_label_mapping[all_labels[idx]]
    probabilities= f"{max(all_probs[idx]):.2f}"
    print(f"Graph: {graph_names[idx]} Correctly Classified as {correct_label_name} with probability {probabilities}")
    for class_idx, prob in enumerate(all_probs[idx]):
        if prob > 0.1:
            class_name = reverse_label_mapping[class_idx]
            print(f"{class_name}: {prob:.2f}")

# #Print prediction probabilities
# for i, (prob, pred) in enumerate(zip(all_probs, all_preds)):
#     print(f"Graph {i}: Prediction={pred}, Probabilities={max(prob):.3f}")

# Compute Confusion Matrix
conf_matrix = confusion_matrix(all_labels, all_preds)

# Print classification report with class names
print("\n📑 Classification Report:")
print(classification_report(all_labels, all_preds, target_names=list(reverse_label_mapping.values()), digits=2, zero_division=0))

# Plot the confusion matrix
plt.figure(figsize=(8, 6))

sorted_labels = [reverse_label_mapping[i] for i in sorted(reverse_label_mapping.keys())]
disp = ConfusionMatrixDisplay(conf_matrix, display_labels=sorted_labels)

#disp = ConfusionMatrixDisplay(conf_matrix, display_labels=list(reverse_label_mapping.values()))
fig, ax = plt.subplots(figsize=(8, 8))  # Adjust figure size
disp.plot(cmap="Blues", values_format="d", ax=ax)  # Pass ax to avoid creating a new figure

# Rotate x-axis labels
plt.xticks(rotation=90)

# Adjust layout to prevent cutoff
plt.tight_layout()
plt.savefig("confusion_matrix.png")
#plt.show()

# # Save predictions and probabilities
# torch.save(all_preds, "test_predictions.pt")
# torch.save(all_probs, "test_probabilities.pt")
