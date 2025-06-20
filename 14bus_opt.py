import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import pandas as pd

# Define model
class TopologyDecisions(nn.Module):
    def __init__(self, input_dim, n_branches,  hidden_dim=128):
        super().__init__()
        self.linear1 = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.linear3 = torch.nn.Linear(hidden_dim, n_branches)

    def forward(self, x):
        x = self.linear1(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation(x)
        x = self.linear3(x)
        return x

# Helper function to add noise
def perturb(tensor, scale=0.10):
    noise = torch.randn_like(tensor) * scale
    return tensor + noise

# Load 'loads' data
data = pd.read_csv('loads.csv')
qd_vals = torch.tensor(data['qd'], dtype=torch.float32)
pd_vals = torch.tensor(data['pd'], dtype=torch.float32)
load_status = torch.tensor(data['status'], dtype=torch.float32)

# Load 'branches' data
data = pd.read_csv('branches.csv')
risk_weight = torch.tensor(data['alpha'][0], dtype=torch.float32)
branch_risks = torch.tensor(data['prisk'], dtype=torch.float32)
branch_status = torch.tensor(data['status'], dtype=torch.float32)

# Format input and record output shape
num_branches = branch_status.shape[0]
x = torch.cat([pd_vals, qd_vals, branch_risks, branch_status, load_status], dim=0)
input_dim = x.shape[0]

# Declare model and optimizer
model = TopologyDecisions(input_dim=input_dim, n_branches=num_branches)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

# Declare loss criteria and target tensor
loss_func = torch.nn.BCEWithLogitsLoss()
target = torch.zeros(num_branches)

# Copy original optimized data to perturb in training
qd_base = qd_vals.clone()
pd_base = pd_vals.clone()
risk_base = branch_risks.clone()

## Train!!
epochs = 200
for i in range(epochs):
    model.train()
    optimizer.zero_grad()

    output = model(x)

    loss = loss_func(output, target)
    penalty = torch.sigmoid(output)
    loss = loss + 0.01 * (penalty * risk_weight).mean() 

    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

# Evaluate final model and output clammped probabilities of branch power
model.eval()
with torch.no_grad():
    ouput = model(x)
    probs = torch.sigmoid(output)
branch_binary = (probs > 0.5).int()

print("Branch decisions (on/off):", branch_binary)

