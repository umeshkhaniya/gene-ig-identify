import torch
import json
import numpy as np
import pandas as pd
import torch.nn.functional as F
from torch_geometric.loader import DataLoader
from torch.nn import Linear, BatchNorm1d, Dropout, ReLU, Sequential
from torch_geometric.nn import GINEConv, global_mean_pool, global_add_pool, global_max_pool
from sklearn.metrics import classification_report, confusion_matrix, ConfusionMatrixDisplay



edge_in_channels_featutes = 8 
node_features  = 1320  

num_igtpyes = 12 



#check: 
test_graphs = torch.load("./uniquegenes_graphs_nocor_humanprot.pt",weights_only=False)
test_labels = torch.load("./uniquegenes_graph_lookup_nocor_humanprot.pt",weights_only=False)
# Define label mapping
label_mapping = {'IgV':0, 'IgC1': 1,  'IgC2':2,'IgI':3, 'IgE':4,'Cadherin': 5, 'IgFN3':6,
    'Lamin':7,  'SOD':8,'IgFN3-like':9, 'CD19':10, "JellyRoll":11} 

reverse_label_mapping = {v: k for k, v in label_mapping.items()}
num_igtpyes = len(label_mapping)

# Print dataset sizes

print(f"Test set: {len(test_graphs)} graphs")



device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load best hyperparameters
with open("../GIN_train/best_hyperparameters.json", "r") as f:
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

        #pool_output_dim = hidden_dim  * num_layers

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
model.load_state_dict(torch.load("../GIN_train/best_graph_model.pth", map_location=device))
model.eval()

test_loader = DataLoader(test_graphs, batch_size=best_params["batch_size"], shuffle=False)


# Evaluation
results = []
with torch.no_grad():
    for data in test_loader:
        data = data.to(device)
        out = model(data)
        probs = F.softmax(out, dim=1).cpu().numpy()
        preds = probs.argmax(axis=1)
        graph_names = data.unique_name_file

        for i in range(len(graph_names)):
            top_label_idx = preds[i]
            top_label = reverse_label_mapping[top_label_idx]
            top_prob = round(probs[i][top_label_idx], 4)

            if top_prob < 0.5:
                top_label = "Other"

            result = {
                "Graph": graph_names[i],
                "Top_Predicted_Label": top_label,
                "Probability": top_prob
            }
            results.append(result)

# Save results
df_results = pd.DataFrame(results)
print(df_results)
df_results.to_excel("top_predictions_uniquegenes.xlsx", index=False)
print(f"✅ Saved {len(df_results)} top predictions to top_predictions.xlsx")
