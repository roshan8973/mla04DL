import torch
import torch.nn as nn
import torch.optim as optim
import pandas as pd
import numpy as np

training = pd.read_csv('u1.base', delimiter='\t', header=None).values
testing = pd.read_csv('u1.test', delimiter='\t', header=None).values
nb_users = int(max(max(training[:,0]), max(testing[:,0])))
nb_movies = int(max(max(training[:,1]), max(testing[:,1])))

def convert(data):
    new_data = []
    for user_id in range(1, nb_users+1):
        movies = data[:,1][data[:,0]==user_id]
        ratings = data[:,2][data[:,0]==user_id]
        r = np.zeros(nb_movies)
        r[movies-1] = ratings
        new_data.append(list(r))
    return torch.FloatTensor(new_data)

train_set = convert(training)
test_set = convert(testing)
train_set[train_set==0] = -1
train_set[train_set==1] = 0
train_set[train_set>=2] = 1
test_set[test_set==0] = -1
test_set[test_set==1] = 0
test_set[test_set>=2] = 1

class RBM():
    def __init__(self, nv, nh):
        self.W = torch.randn(nh, nv)
        self.a = torch.randn(1, nh)
        self.b = torch.randn(1, nv)
    def sample_h(self, x):
        wx = torch.mm(x, self.W.t()) + self.a
        return torch.sigmoid(wx), torch.bernoulli(torch.sigmoid(wx))
    def sample_v(self, y):
        wy = torch.mm(y, self.W) + self.b
        return torch.sigmoid(wy), torch.bernoulli(torch.sigmoid(wy))
    def train(self, v0, vk, ph0, phk):
        self.W += torch.mm(ph0.t(), v0) - torch.mm(phk.t(), vk)
        self.b += torch.sum((v0 - vk), 0)
        self.a += torch.sum((ph0 - phk), 0)

nv = nb_movies
nh = 100
batch_size = 100
rbm = RBM(nv, nh)

for epoch in range(10):
    train_loss = 0
    for i in range(0, nb_users - batch_size, batch_size):
        v0 = train_set[i:i+batch_size]
        vk = train_set[i:i+batch_size]
        ph0,_ = rbm.sample_h(v0)
        for k in range(10):
            _,hk = rbm.sample_h(vk)
            _,vk = rbm.sample_v(hk)
            vk[v0==-1] = v0[v0==-1]
        phk,_ = rbm.sample_h(vk)
        rbm.train(v0, vk, ph0, phk)
        train_loss += torch.mean(torch.abs(v0[v0!=-1] - vk[v0!=-1]))
    print(f'Epoch {epoch+1} Loss: {train_loss:.4f}')

test_loss = 0
for user in range(nb_users):
    v = train_set[user:user+1]
    vt = test_set[user:user+1]
    if len(vt[vt!=-1]) > 0:
        _,h = rbm.sample_h(v)
        _,v_pred = rbm.sample_v(h)
        test_loss += torch.mean(torch.abs(vt[vt!=-1] - v_pred[vt!=-1]))
print(f'Test Loss: {test_loss:.4f}')
