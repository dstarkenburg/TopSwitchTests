# Using assistance from https://www.youtube.com/watch?v=OMDn66kM9Qc
import torch
from torch import nn
from torch import optim
from torchvision import datasets, transforms
from torch.utils.data import random_split, DataLoader

torch.randn(5).cuda()

train_data = datasets.MNIST('data', train=True, download=True, transform=transforms.ToTensor())
train, val = random_split(train_data, [55000, 5000])
train_loader = DataLoader(train, batch_size=32)
val_loader = DataLoader(val, batch_size=32)

model = nn.Sequential(
    nn.Linear(28*28, 784),
    nn.Sigmoid(),
    nn.Linear(784, 400),
    nn.Sigmoid(),
    nn.Linear(400, 400),
    nn.Sigmoid(),
    nn.Linear(400, 10)
)

optimizer = optim.Adam(model.parameters(), lr=1e-4)

loss = nn.CrossEntropyLoss()

num_epochs = 10
for epoch in range(num_epochs):
    losses = list()
    for batch in train_loader:
        x, y = batch

        b = x.size(0)
        x = x.view(b, -1)

        l = model(x)

        j = loss(l, y)

        model.zero_grad()

        j.backward()

        optimizer.step()

        losses.append(j.item())

    print(f'Epoch {epoch + 1},  training loss: {torch.tensor(losses).mean():.2f}')

    losses = list()
    for batch in val_loader:
        x, y = batch
        
        b = x.size(0)
        x = x.view(b, -1)

        with torch.no_grad():
            l = model(x)

        j = loss(l, y)

        losses.append(j.item())

    print(f'Epoch {epoch + 1},  validation loss: {torch.tensor(losses).mean():.2f}')