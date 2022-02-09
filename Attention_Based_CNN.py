# -*- coding: utf-8 -*-
import torch
from torch import nn
import torch.nn.functional as F
import numpy as np
import pandas as pd
from torch.utils.data import Dataset
from torch.utils.data import DataLoader

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
criterion = nn.CrossEntropyLoss()

# BCI III Dataset V
class DS(Dataset):
  def __init__(self,fn):
    self.data = pd.read_csv(fn + "_DATA.csv" ,header=None).to_numpy()
    self.label = pd.read_csv(fn + "_LABEL.csv" ,header=None).to_numpy()

  def __len__(self):
    return len(self.data)-7

  def __getitem__(self,idx):
    return self.data[idx:idx+8],self.label[idx+7]

ds_train = []
for session in range(1,4):
  ds_train.append(DS("train/train_subject1_psd0" + str(session)))
dataloader_train = DataLoader(torch.utils.data.ConcatDataset(ds_train), batch_size=64, shuffle=True)
ds_tests = []
ds_test = DS("test/test_subject1_psd04")
dataloader_test = DataLoader(ds_test, batch_size=32, shuffle=True)

#@title Attention + CNN
class CNN(nn.Module):
    def __init__(self):
      super(CNN, self).__init__()
      self.conv1 = nn.Conv2d(1,3, (3,7))
      self.bn1 = nn.BatchNorm2d(3, False)
      
      self.conv2 = nn.Conv2d(3, 3, (3,7))
      self.bn2 = nn.BatchNorm2d(3, False)

      self.conv3 = nn.Conv2d(3, 5, (3,7))
      self.bn3 = nn.BatchNorm2d(5, False)

      self.fc1 = nn.Linear(160, 40)
      self.fc2 = nn.Linear(40, 3)
      self.pool = nn.MaxPool2d((1,2))

    def forward(self, x):
      x = F.relu(self.conv1(x))
      x = self.pool(self.bn1(F.dropout(x, 0.25)))

      x = F.relu(self.conv2(x))
      x = self.bn2(F.dropout(x, 0.25))

      x = F.relu(self.conv3(x))
      x = self.pool(self.bn3(F.dropout(x, 0.25)))

      x = torch.flatten(x, 1)
      x = F.relu(self.fc1(x))
      x = F.relu(self.fc2(x))
      return x

class Attention_Temp(nn.Module):
  def __init__(self):
    super(Attention_Temp, self).__init__()
    self.position = nn.Linear(8,8)
    self.values = nn.Linear(96,96, bias=False)
    self.keys = nn.Linear(96,96, bias=False)
    self.queries = nn.Linear(96,96, bias=False)
    
  def forward(self, x):
    x = x.squeeze(1)
    p = torch.arange(8).to(device)
    embed = x+self.position(p.float()).view(8,1)
    values=keys=queries=embed
    values,keys,queries = self.values(values.float()),self.keys(keys.float()),self.queries(queries.float())  
    attention = torch.softmax(torch.einsum("bqx,bky->bxy", [queries,keys]),dim=1)
    x = torch.einsum("bxy,bvn->bvy", [attention,values])
    return x

class Attention_Spec(nn.Module):
  def __init__(self):
    super(Attention_Spec, self).__init__()
    self.position = nn.Linear(96,96)
    self.values = nn.Linear(8,8, bias=False)
    self.keys = nn.Linear(8,8, bias=False)
    self.queries = nn.Linear(8,8, bias=False)
    
  def forward(self, x):
    x = x.squeeze(1)
    x = torch.transpose(x,1,2)
    p = torch.arange(96).to(device)
    embed = x+self.position(p.float()).view(96,1)
    values=keys=queries=embed
    values,keys,queries = self.values(values.float()),self.keys(keys.float()),self.queries(queries.float())  
    attention = torch.softmax(torch.einsum("bqx,bky->bxy", [queries,keys]),dim=1)
    x = torch.einsum("bxy,bvn->bvy", [attention,values])
    return torch.transpose(x,1,2)

cnn = CNN()
att_temp = Attention_Temp()
att_spec = Attention_Spec()

class Att_CNN(nn.Module):
  def __init__(self, att, att2, cnn): 
      super(Att_CNN, self).__init__()
      self.att = att
      self.att2 = att2
      self.fc1 = nn.Linear(20,20)
      self.fc2 = nn.Linear(20,20)
      self.cnn = cnn

  def forward(self, x):
      att1 = self.att(x.squeeze(1))
      att2 = self.att2(x.squeeze(1))
      x = self.fc1(att1).unsqueeze(1) + self.fc2(att2).unsqueeze(1) + x
      out = self.cnn(x)
      return out

att_cnn = Att_CNN(att_temp, att_spec, cnn).to(device)
optimizer = torch.optim.Adam(att_cnn.parameters(), lr=0.0005,  betas=(0.9, 0.98), eps=1e-7)

# training
for epoch in range(100):
  att_cnn.train()
  for i,(X, y) in enumerate(dataloader_train):
    X, y = X.to(device), y.to(device) 
    optimizer.zero_grad()
    pred = att_cnn(X.unsqueeze(1).float())
    loss = criterion(pred, y)
    loss.backward()
    optimizer.step()

  att_cnn.eval()
  correct = 0
  for i,(X, y) in enumerate(dataloader_test):
    X, y = X.to(device), y.to(device)
    with torch.no_grad():
      pred = att_cnn(X.unsqueeze(1).float())
      correct += (pred.argmax(1) == y).type(torch.float).sum().item()
  if epoch % 10==0:
    print('\nTest Epoch: {}, Accuracy: {}/{} ({:.0f}%)'.format(
        epoch+1, correct, len(dataloader_test.dataset),
        100. * correct / len(dataloader_test.dataset)))


#@title CBAM
### https://blog.csdn.net/weixin_38241876/article/details/109853433
class ChannelAttentionModule(nn.Module):
    def __init__(self, channel, ratio=4):
        super(ChannelAttentionModule, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.max_pool = nn.AdaptiveMaxPool2d(1)
 
        self.shared_MLP = nn.Sequential(
            nn.Conv2d(channel, channel // ratio, 1, bias=False),
            nn.ReLU(),
            nn.Conv2d(channel // ratio, channel, 1, bias=False)
        )
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avgout = self.shared_MLP(self.avg_pool(x))
        maxout = self.shared_MLP(self.max_pool(x))
        return self.sigmoid(avgout + maxout)
 
class SpatialAttentionModule(nn.Module):
    def __init__(self):
        super(SpatialAttentionModule, self).__init__()
        self.conv2d = nn.Conv2d(in_channels=2, out_channels=1, kernel_size=7, stride=1, padding=3)
        self.sigmoid = nn.Sigmoid()
 
    def forward(self, x):
        avgout = torch.mean(x, dim=1, keepdim=True)
        maxout, _ = torch.max(x, dim=1, keepdim=True)
        out = torch.cat([avgout, maxout], dim=1)
        out = self.sigmoid(self.conv2d(out))
        return out
 
class CBAM_Module(nn.Module):
    def __init__(self, channel):
        super(CBAM_Module, self).__init__()
        self.channel_attention = ChannelAttentionModule(channel)
        self.spatial_attention = SpatialAttentionModule()
 
    def forward(self, x):
        out = self.channel_attention(x) * x
        out = self.spatial_attention(out) * out
        return out


class CBAM(nn.Module):
    def __init__(self):
        super(CBAM, self).__init__()
        self.conv1 = CBAM_Module(20)
        self.bn1 = nn.BatchNorm2d(20, False)
        
        self.conv2 = CBAM_Module(20)
        self.bn2 = nn.BatchNorm2d(20, False)

        self.conv3 = CBAM_Module(20)
        self.bn3 = nn.BatchNorm2d(20, False)

        self.fc1 = nn.Linear(20, 20)
        self.fc2 = nn.Linear(5, 5)
        self.fc3 = nn.Linear(100, 10)
        self.fc4 = nn.Linear(10, 5)
        self.pool = nn.MaxPool2d((1,2))

    def forward(self, x):
      x = self.fc1(torch.transpose(x,1,2).squeeze(3))
      x = self.fc2(torch.transpose(x,1,2)).unsqueeze(3)
      x = F.relu(self.conv1(x))
      x = self.bn1(F.dropout(x, 0.25))

      x = F.relu(self.conv2(x))
      x = self.bn2(F.dropout(x, 0.25))

      x = F.relu(self.conv3(x))
      x = self.bn3(F.dropout(x, 0.25))

      x = torch.flatten(x, 1)
      x = F.relu(self.fc3(x))
      x = F.relu(self.fc4(x))
      return x

cbam = CBAM().to(device)
optimizer = torch.optim.Adam(cbam.parameters(), lr=0.0005, betas=(0.9, 0.98), eps=1e-9)