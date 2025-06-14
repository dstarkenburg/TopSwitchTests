import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import pandas as pd

class TopologyDecisions(nn.Module):
    def __init__(self, input_dim, n_branches, n_loads, hidden_dim=128):
        super().__init__()
        self.shared = nn.Sequential(
            nn.Linear(input_dim, hidden_dim),
            nn.Sigmoid(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.Sigmoid())
        self.branch_head = nn.Sequential(
            nn.Linear(hidden_dim, n_branches),
            nn.Sigmoid() )
        self.load_head = nn.Sequential(
            nn.Linear(hidden_dim, n_loads),
            nn.Sigmoid())
        
    def forward(self, x):
        shared = self.shared(x)
        branches = self.branch_head(shared)
        loads = self.load_head(shared)
        return branches, loads
    
def custom_loss(branch_status, load_status, branch_risks, pd, qd, risk_weight, alpha=1.0):
    risk = (branch_status * branch_risks).sum(dim=0) * risk_weight.squeeze()
    total_power = pd + qd
    load_served = (load_status * total_power).sum(dim=0)
    loss = (risk - alpha * load_served).mean()
    return loss

data = pd.read_csv('loads.csv')
qd_vals = torch.tensor(data['qd'], dtype=torch.float32)
pd_vals = torch.tensor(data['pd'], dtype=torch.float32)
load_status = torch.tensor(data['status'], dtype=torch.float32)

data = pd.read_csv('branches.csv')
risk_weight = torch.tensor(data['alpha'][0], dtype=torch.float32)
branch_risks = torch.tensor(data['prisk'], dtype=torch.float32)
branch_status = torch.tensor(data['status'], dtype=torch.float32)

x = torch.cat([pd_vals, qd_vals, branch_risks, branch_status, load_status], dim=0)
num_loads = load_status.shape[0]
num_branches = branch_status.shape[0]
risk_weight = risk_weight.repeat(1, num_branches)
input_dim = x.shape[0]


model = TopologyDecisions(input_dim=input_dim, n_branches=num_branches, n_loads=num_loads)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

## Train!!

epochs = 1500

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    branch_status, load_status = model(x)

    loss = custom_loss(
        branch_status=branch_status,
        load_status=load_status,
        branch_risks=branch_risks,
        pd=pd_vals,
        qd=qd_vals,
        risk_weight=risk_weight
    )

    loss.backward()
    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

model.eval()
with torch.no_grad():
    branch_status, load_status = model(x)
branch_binary = (branch_status > 0.5).int()
load_binary = (load_status > 0.5).int()

print("Branch decisions (on/off):", branch_binary)
print("Load decisions (on/off):", load_binary)