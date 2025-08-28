import torch,torch.nn as nn,torch.optim as optim
from torchvision import datasets,transforms
from torch.utils.data import DataLoader
import matplotlib.pyplot as plt

class DAE(nn.Module):
    def __init__(self):
        super().__init__()
        self.encoder=nn.Sequential(nn.Linear(784,128),nn.ReLU(),nn.Linear(128,64))
        self.decoder=nn.Sequential(nn.Linear(64,128),nn.ReLU(),nn.Linear(128,784),nn.Sigmoid())
    def forward(self,x): return self.decoder(self.encoder(x))

transform=transforms.Compose([transforms.ToTensor()])
train_data=datasets.MNIST(root='./data',train=True,download=True,transform=transform)
train_loader=DataLoader(train_data,batch_size=128,shuffle=True)

model=DAE()
criterion=nn.MSELoss()
optimizer=optim.Adam(model.parameters(),lr=1e-3)

for epoch in range(5):
    loss=0
    for img,_ in train_loader:
        noisy=img+0.5*torch.randn_like(img)
        noisy=torch.clamp(noisy,0.,1.).view(-1,784)
        clean=img.view(-1,784)
        output=model(noisy)
        l=criterion(output,clean)
        optimizer.zero_grad(); l.backward(); optimizer.step()
        loss+=l.item()
    print(f'Epoch {epoch+1}, Loss: {loss/len(train_loader):.4f}')
