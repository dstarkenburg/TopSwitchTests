import torch
import h5py
import numpy
import matplotlib.pyplot as plt
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader
import pandas as pd

# Define model
class TopologyDecisions(nn.Module):
    def __init__(self, input_dim, n_branches=20,  hidden_dim=43):
        super().__init__()
        self.linear = torch.nn.Linear(input_dim, hidden_dim)
        self.activation = torch.nn.Sigmoid()
        self.linear2 = torch.nn.Linear(hidden_dim, hidden_dim)
        self.activation2 = torch.nn.Sigmoid()
        self.linear3= torch.nn.Linear(hidden_dim, n_branches)

    def forward(self, x):
        x = self.linear(x)
        x = self.activation(x)
        x = self.linear2(x)
        x = self.activation2(x)
        x = self.linear3(x)
        return x
    
# Import and format data
filename = "data.h5"
x_train_temp = []
y_train_temp = []
with h5py.File(filename, "r") as f:
    for i in f["train_data"].keys():
        temp_power_risk = torch.tensor(numpy.array(f["train_data"][i]["branch"]["power_risk"][()]), dtype=torch.float32)
        temp_qd = torch.tensor(numpy.array(f["train_data"][i]["load"]["qd"][()]), dtype=torch.float32)
        temp_qd = torch.tensor(numpy.array(f["train_data"][i]["load"]["pd"][()]), dtype=torch.float32)
        temp_alpha = torch.tensor(numpy.array(f["train_data"][i]["alpha"][()]), dtype=torch.float32)
        x = torch.cat([temp_power_risk, temp_qd, temp_qd, temp_alpha], dim = 0)
        y = torch.tensor(numpy.array(f["train_data"][i]["branch"]["status"][()]), dtype=torch.float32)
        x_train_temp.append(x)
        y_train_temp.append(y)

x_test_temp = []
y_test_temp = []
with h5py.File(filename, "r") as f:
    for i in f["test_data"].keys():
        temp_power_risk = torch.tensor(numpy.array(f["test_data"][i]["branch"]["power_risk"][()]), dtype=torch.float32)
        temp_qd = torch.tensor(numpy.array(f["test_data"][i]["load"]["qd"][()]), dtype=torch.float32)
        temp_qd = torch.tensor(numpy.array(f["test_data"][i]["load"]["pd"][()]), dtype=torch.float32)
        temp_alpha = torch.tensor(numpy.array(f["test_data"][i]["alpha"][()]), dtype=torch.float32)
        x = torch.cat([temp_power_risk, temp_qd, temp_qd, temp_alpha], dim = 0)
        y = torch.tensor(numpy.array(f["test_data"][i]["branch"]["status"][()]), dtype=torch.float32)
        x_test_temp.append(x)
        y_test_temp.append(y)

x_val_temp = []
y_val_temp = []
with h5py.File(filename, "r") as f:
    for i in f["val_data"].keys():
        temp_power_risk = torch.tensor(numpy.array(f["val_data"][i]["branch"]["power_risk"][()]), dtype=torch.float32)
        temp_qd = torch.tensor(numpy.array(f["val_data"][i]["load"]["qd"][()]), dtype=torch.float32)
        temp_qd = torch.tensor(numpy.array(f["val_data"][i]["load"]["pd"][()]), dtype=torch.float32)
        temp_alpha = torch.tensor(numpy.array(f["val_data"][i]["alpha"][()]), dtype=torch.float32)
        x = torch.cat([temp_power_risk, temp_qd, temp_qd, temp_alpha], dim = 0)
        y = torch.tensor(numpy.array(f["val_data"][i]["branch"]["status"][()]), dtype=torch.float32)
        x_val_temp.append(x)
        y_val_temp.append(y)

class OpsDataset(torch.utils.data.Dataset):
    def __init__(self, inputs, truths):
        self.inputs = inputs
        self.truths = truths

    def __len__(self):
        return len(self.truths)

    def __getitem__(self, idx):
        return self.inputs[idx], self.truths[idx]

num_branches = len(y_test_temp[1])
input_dim = len(x_test_temp[1])

train_dataset = OpsDataset(x_train_temp, y_train_temp)
test_dataset = OpsDataset(x_test_temp, y_test_temp)
val_dataset = OpsDataset(x_val_temp, y_val_temp)

model = TopologyDecisions(input_dim=input_dim, n_branches=num_branches)
optimizer = torch.optim.Adam(model.parameters(), lr=1e-4)

train_dataloader = torch.utils.data.DataLoader(train_dataset, batch_size = 50)
test_dataloader = torch.utils.data.DataLoader(test_dataset, batch_size = 50)
val_dataloader = torch.utils.data.DataLoader(val_dataset, batch_size = 50)

train_losses = []
val_losses = []
epochs = 75
criterion = nn.BCEWithLogitsLoss()
for i in range(epochs):
    model.train()
    train_loss = 0.0
    for input, truth in train_dataloader:
        if torch.cuda.is_available():
            input, truth = input.cuda(), truth.cuda()
            model.cuda()
        else:
            model.cpu()

        optimizer.zero_grad()
    
        outputs = model(input)
        loss = criterion(outputs, truth)

        loss.backward()
        optimizer.step()

        train_loss += loss.item()
    train_losses.append(train_loss / len(train_dataloader))  

    model.eval()
    val_loss = 0.0
    for input, truth in val_dataloader:
        if torch.cuda.is_available():
            input, truth = input.cuda(), truth.cuda()
        
        output = model(input)
        loss = criterion(outputs, truth)

        val_loss += loss.item()
    val_losses.append(val_loss / len(val_dataloader))  
    #if i % 5 == 0:
    print(f"[Epoch {i+1}] Training loss: {train_loss / len(train_dataloader):.2f}, Validation loss: {val_loss / len(val_dataloader):.2f}")


test_scores = []
with torch.no_grad():
    for input, truth in test_dataloader:
        if torch.cuda.is_available():
            input, truth = input.cuda(), truth.cuda()
        outputs = model(input)
        clamp = torch.round(outputs)
        
        correct = 0
        truth_size = len(truth)
        for e, i in enumerate(clamp[0]):
            if i == truth[0][e]:
                correct += correct
        test_scores.append(correct / truth_size)
    print(f"Accuracy Average: {sum(test_scores) / len(test_scores)}")
        
        


plt.plot(train_losses, label="Training Loss")
plt.plot(val_losses, label="Validation Loss")
plt.title('Loss Curves')
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.legend()
plt.show()


