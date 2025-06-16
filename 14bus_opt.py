import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import pandas as pd

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
    

data = pd.read_csv('loads.csv')
qd_vals = torch.tensor(data['qd'], dtype=torch.float32)
pd_vals = torch.tensor(data['pd'], dtype=torch.float32)
load_status = torch.tensor(data['status'], dtype=torch.float32)

data = pd.read_csv('branches.csv')
risk_weight = torch.tensor(data['alpha'][0], dtype=torch.float32)
branch_risks = torch.tensor(data['prisk'], dtype=torch.float32)
branch_status = torch.tensor(data['status'], dtype=torch.float32)

num_branches = branch_status.shape[0]
x = torch.cat([pd_vals, qd_vals, branch_risks, branch_status, load_status], dim=0)
input_dim = x.shape[0]


model = TopologyDecisions(input_dim=input_dim, n_branches=num_branches)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

weight = risk_weight.repeat(num_branches)
loss_func = torch.nn.BCEWithLogitsLoss()
target = torch.zeros(num_branches)
## Train!!

epochs = 1500

for epoch in range(epochs):
    model.train()

    optimizer.zero_grad()

    output = model(x)

    loss = loss_func(output, target)

    penalty = torch.sigmoid(output)
    loss = loss + 0.01 * penalty.mean() 

    optimizer.step()

    if epoch % 10 == 0:
        print(f"Epoch {epoch}: Loss = {loss.item():.4f}")

model.eval()
with torch.no_grad():
    ouput = model(x)
branch_binary = (output > 0.5).int()

print("Branch decisions (on/off):", branch_binary)