# -*- coding: utf-8 -*-
"""
Created on Sat Mar 12 13:49:35 2022

@author: Edouard LANSIAUX
"""

import os
import torch
from torch import nn
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import pylab as pl

from torch.utils.data import TensorDataset, DataLoader
from sklearn.metrics import mean_squared_error, mean_absolute_error
   
    
class IMVTensorLSTM(torch.jit.ScriptModule):
    
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_i = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_f = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.U_o = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_i = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_f = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.W_o = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_i = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_f = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.b_o = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
    
    @torch.jit.script_method
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cpu()
        c_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cpu()
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            # eq 5
            i_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_i) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_i) + self.b_i)
            f_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_f) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_f) + self.b_f)
            o_tilda_t = torch.sigmoid(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_o) + \
                                torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_o) + self.b_o)
            # eq 6
            c_tilda_t = c_tilda_t*f_tilda_t + i_tilda_t*j_tilda_t
            # eq 7
            h_tilda_t = (o_tilda_t*torch.tanh(c_tilda_t))
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)
        
        return mean, alphas, betas

    
class IMVFullLSTM(torch.jit.ScriptModule):
    __constants__ = ["n_units", "input_dim"]
    def __init__(self, input_dim, output_dim, n_units, init_std=0.02):
        super().__init__()
        self.U_j = nn.Parameter(torch.randn(input_dim, 1, n_units)*init_std)
        self.W_j = nn.Parameter(torch.randn(input_dim, n_units, n_units)*init_std)
        self.b_j = nn.Parameter(torch.randn(input_dim, n_units)*init_std)
        self.W_i = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.W_f = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.W_o = nn.Linear(input_dim*(n_units+1), input_dim*n_units)
        self.F_alpha_n = nn.Parameter(torch.randn(input_dim, n_units, 1)*init_std)
        self.F_alpha_n_b = nn.Parameter(torch.randn(input_dim, 1)*init_std)
        self.F_beta = nn.Linear(2*n_units, 1)
        self.Phi = nn.Linear(2*n_units, output_dim)
        self.n_units = n_units
        self.input_dim = input_dim
    @torch.jit.script_method
    def forward(self, x):
        h_tilda_t = torch.zeros(x.shape[0], self.input_dim, self.n_units).cpu()
        c_t = torch.zeros(x.shape[0], self.input_dim*self.n_units).cpu()
        outputs = torch.jit.annotate(List[Tensor], [])
        for t in range(x.shape[1]):
            # eq 1
            j_tilda_t = torch.tanh(torch.einsum("bij,ijk->bik", h_tilda_t, self.W_j) + \
                                   torch.einsum("bij,jik->bjk", x[:,t,:].unsqueeze(1), self.U_j) + self.b_j)
            inp =  torch.cat([x[:, t, :], h_tilda_t.view(h_tilda_t.shape[0], -1)], dim=1)
            # eq 2
            i_t = torch.sigmoid(self.W_i(inp))
            f_t = torch.sigmoid(self.W_f(inp))
            o_t = torch.sigmoid(self.W_o(inp))
            # eq 3
            c_t = c_t*f_t + i_t*j_tilda_t.view(j_tilda_t.shape[0], -1)
            # eq 4
            h_tilda_t = (o_t*torch.tanh(c_t)).view(h_tilda_t.shape[0], self.input_dim, self.n_units)
            outputs += [h_tilda_t]
        outputs = torch.stack(outputs)
        outputs = outputs.permute(1, 0, 2, 3)
        # eq 8
        alphas = torch.tanh(torch.einsum("btij,ijk->btik", outputs, self.F_alpha_n) +self.F_alpha_n_b)
        alphas = torch.exp(alphas)
        alphas = alphas/torch.sum(alphas, dim=1, keepdim=True)
        g_n = torch.sum(alphas*outputs, dim=1)
        hg = torch.cat([g_n, h_tilda_t], dim=2)
        mu = self.Phi(hg)
        betas = torch.tanh(self.F_beta(hg))
        betas = torch.exp(betas)
        betas = betas/torch.sum(betas, dim=1, keepdim=True)
        mean = torch.sum(betas*mu, dim=1)
        return mean, alphas, betas


data1 =  pd.read_csv("C:/Users/PATRICE LANSIAUX/Desktop/crypto/papers/doge VS LTC/data/DOGE_final.csv")
data2 =  pd.read_csv("C:/Users/PATRICE LANSIAUX/Desktop/crypto/papers/doge VS LTC/data/LTC_final.csv") 

del data1['date']
del data1['transactions']
del data1['median_transaction_value']
del data1['av_transaction_value']
del data1['market_cap']
del data1['top_100_percent']
del data1['active_addresses']
data1 = data1[['tweets', 'price']]
del data2['date']
del data2['transactions']
del data2['median_transaction_value']
del data2['av_transaction_value']
del data2['market_cap']
del data2['top_200_percent']
del data2['active_addresses']
data2 = data2[['tweets', 'price']]

batch_size = 7
timesteps = 20
n_timeseries1 = data1.shape[1]
n_timeseries2 = data2.shape[1]
train_length = 2162
val_length = 145
test_length = 145
target = "price"

X1 = np.zeros((len(data1), timesteps, data1.shape[1]))
X2 = np.zeros((len(data2), timesteps, data2.shape[1]))

for i, name in enumerate(list(data1.columns)):
    print(name)
    for j in range(timesteps):
        X1[:, j, i] = data1[name].shift(timesteps - j - 1).fillna(method="bfill")

for i, name in enumerate(list(data2.columns)):
    print(name)
    for j in range(timesteps):
        X2[:, j, i] = data2[name].shift(timesteps - j - 1).fillna(method="bfill")


prediction_horizon = 1
target1 = data1["price"].shift(-prediction_horizon).fillna(method="ffill").values
target2 = data2["price"].shift(-prediction_horizon).fillna(method="ffill").values

X1 = X1[timesteps:]
target1 = target1[timesteps:]
X2 = X2[timesteps:]
target2 = target2[timesteps:]

X_train1 = X1[:train_length]
X_val1 = X1[train_length:train_length+val_length]
X_test1 = X1[-val_length:]
target_train1 = target1[:train_length]
target_val1 = target1[train_length:train_length+val_length]
target_test1 = target1[-val_length:]
X_train2 = X2[:train_length]
X_val2 = X2[train_length:train_length+val_length]
X_test2 = X2[-val_length:]
target_train2 = target2[:train_length]
target_val2 = target2[train_length:train_length+val_length]
target_test2 = target2[-val_length:]

X_train_max1 = X_train1.max(axis=0)
X_train_min1 = X_train1.min(axis=0)
target_train_max1 = target_train1.max(axis=0)
target_train_min1 = target_train1.min(axis=0)
X_train_max2 = X_train2.max(axis=0)
X_train_min2 = X_train2.min(axis=0)
target_train_max2 = target_train2.max(axis=0)
target_train_min2 = target_train2.min(axis=0)

X_train1 = (X_train1 - X_train_min1) / (X_train_max1 - X_train_min1)
X_val1 = (X_val1 - X_train_min1) / (X_train_max1 - X_train_min1)
X_test1 = (X_test1 - X_train_min1) / (X_train_max1 - X_train_min1)

target_train1 = (target_train1 - target_train_min1) / (target_train_max1 - target_train_min1)
target_val1 = (target_val1 - target_train_min1) / (target_train_max1 - target_train_min1)
target_test1 = (target_test1 - target_train_min1) / (target_train_max1 - target_train_min1)

X_train2 = (X_train2 - X_train_min2) / (X_train_max2 - X_train_min2)
X_val2 = (X_val2 - X_train_min2) / (X_train_max2 - X_train_min2)
X_test2 = (X_test2 - X_train_min2) / (X_train_max2 - X_train_min2)

target_train2 = (target_train2 - target_train_min2) / (target_train_max2 - target_train_min2)
target_val2 = (target_val2 - target_train_min2) / (target_train_max2 - target_train_min2)
target_test2 = (target_test2 - target_train_min2) / (target_train_max2 - target_train_min2)

X_train_t1 = torch.Tensor(X_train1)
X_val_t1 = torch.Tensor(X_val1)
X_test_t1 = torch.Tensor(X_test1)
target_train_t1 = torch.Tensor(target_train1)
target_val_t1 = torch.Tensor(target_val1)
target_test_t1 = torch.Tensor(target_test1)

X_train_t2 = torch.Tensor(X_train2)
X_val_t2 = torch.Tensor(X_val2)
X_test_t2 = torch.Tensor(X_test2)
target_train_t2 = torch.Tensor(target_train2)
target_val_t2 = torch.Tensor(target_val2)
target_test_t2 = torch.Tensor(target_test2)

model1 = IMVTensorLSTM(X_train_t1.shape[2], 1, 7).cpu()
model2 = IMVTensorLSTM(X_train_t2.shape[2], 1, 7).cpu()

opt1 = torch.optim.Adam(model1.parameters(), lr=0.001)
opt2 = torch.optim.Adam(model2.parameters(), lr=0.001)

epoch_scheduler1 = torch.optim.lr_scheduler.StepLR(opt1, 1, gamma=0.9)
epoch_scheduler2 = torch.optim.lr_scheduler.StepLR(opt2, 1, gamma=0.9)

data_train_loader1 = DataLoader(TensorDataset(X_train_t1, target_train_t1), shuffle=True, batch_size=128)
data_val_loader1 = DataLoader(TensorDataset(X_val_t1, target_val_t1), shuffle=False, batch_size=128)
data_test_loader1 = DataLoader(TensorDataset(X_test_t1, target_test_t1), shuffle=False, batch_size=128)

data_train_loader2 = DataLoader(TensorDataset(X_train_t2, target_train_t2), shuffle=True, batch_size=228)
data_val_loader2 = DataLoader(TensorDataset(X_val_t2, target_val_t2), shuffle=False, batch_size=228)
data_test_loader2 = DataLoader(TensorDataset(X_test_t2, target_test_t2), shuffle=False, batch_size=228)

epochs = 1000
loss = nn.MSELoss()
patience = 35
min_val_loss = 9999
counter = 0


##DOGE
for i in range(epochs):
    mse_train = 0
    for batch_x, batch_y in data_train_loader1:
        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()
        opt1.zero_grad()
        y_pred, alphas, betas = model1(batch_x)
        y_pred = y_pred.squeeze(1)
        l = loss(y_pred, batch_y)
        l.backward()
        mse_train += l.item()*batch_x.shape[0]
        opt1.step()
    epoch_scheduler1.step()
    with torch.no_grad():
        mse_val = 0
        preds1 = []
        true = []
        for batch_x, batch_y in data_val_loader1:
            batch_x = batch_x.cpu()
            batch_y = batch_y.cpu()
            output1, alphas, betas = model1(batch_x)
            output1 = output1.squeeze(1)
            preds1.append(output1.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            mse_val += loss(output1, batch_y).item()*batch_x.shape[0]
    preds1 = np.concatenate(preds1)
    true = np.concatenate(true)
    
    if min_val_loss > mse_val**0.5:
        min_val_loss = mse_val**0.5
        print("Saving...")
        torch.save(model1.state_dict(), "imv_tensor_lstm_nasdaq.pt")
        counter = 0
    else: 
        counter += 1
    
    if counter == patience:
        break
    print("Iter: ", i, "train: ", (mse_train/len(X_train_t1))**0.5, "val: ", (mse_val/len(X_val_t1))**0.5)
    if(i % 10 == 0):
        preds1 = preds1*(target_train_max1 - target_train_min1) + target_train_min1
        true = true*(target_train_max1 - target_train_min1) + target_train_min1
        mse = mean_squared_error(true, preds1)
        mae = mean_absolute_error(true, preds1)
        print("lr: ", opt1.param_groups[0]["lr"])
        print("mse: ", mse, "mae: ", mae)
        pl.figure(figsize=(20, 10))
        plt.plot(preds1)
        plt.plot(true)
        plt.show()

with torch.no_grad():
    mse_val = 0
    preds1 = []
    true1 = []
    alphas = []
    betas = []
    for batch_x, batch_y in data_test_loader1:
        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()
        output, a, b = model1(batch_x)
        output = output.squeeze(1)
        preds1.append(output.detach().cpu().numpy())
        true1.append(batch_y.detach().cpu().numpy())
        alphas.append(a.detach().cpu().numpy())
        betas.append(b.detach().cpu().numpy())
        mse_val += loss(output, batch_y).item()*batch_x.shape[0]
preds1 = np.concatenate(preds1)
true1 = np.concatenate(true1)

preds1 = preds1*(target_train_max1 - target_train_min1) + target_train_min1
true1 = true1*(target_train_max1 - target_train_min1) + target_train_min1

mse = mean_squared_error(true1, preds1)
mae = mean_absolute_error(true1, preds1)

pl.figure(figsize=(20, 10))
plt.plot(preds1)
plt.plot(true1)
plt.show()


alphas = np.concatenate(alphas)
betas = np.concatenate(betas)

alphas = alphas.mean(axis=0)
betas = betas.mean(axis=0)

alphas = alphas[..., 0]
betas = betas[..., 0]

alphas = alphas.transpose(1, 0)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(alphas)
ax.set_xticks(np.arange(X_train_t1.shape[1]))
ax.set_yticks(np.arange(len(data1.columns)))
ax.set_xticklabels(["t-"+str(i) for i in np.arange(X_train_t1.shape[1], 1, 1)])
ax.set_yticklabels(list(data1.columns))
for i in range(len(data1.columns)):
    for j in range(X_train_t1.shape[1]):
        text = ax.text(j, i, round(alphas[i, j], 3),
                       ha="center", va="center", color="w")
ax.set_title("Importance of features and timesteps for DOGECOIN")
#fig.tight_layout()
plt.show()


plt.figure(figsize=(10, 10))
plt.title("Feature importance for DOGECOIN")
plt.bar(range(len(data1.columns)), betas)
plt.xticks(ticks=range(len(data1.columns)), labels=list(data1.columns), rotation=90)

#LTC
for i in range(epochs):
    mse_train = 0
    for batch_x, batch_y in data_train_loader2:
        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()
        opt2.zero_grad()
        y_pred, alphas, betas = model2(batch_x)
        y_pred = y_pred.squeeze(1)
        l = loss(y_pred, batch_y)
        l.backward()
        mse_train += l.item()*batch_x.shape[0]
        opt2.step()
    epoch_scheduler2.step()
    with torch.no_grad():
        mse_val = 0
        preds2 = []
        true = []
        for batch_x, batch_y in data_val_loader2:
            batch_x = batch_x.cpu()
            batch_y = batch_y.cpu()
            output2, alphas, betas = model2(batch_x)
            output2 = output2.squeeze(1)
            preds2.append(output2.detach().cpu().numpy())
            true.append(batch_y.detach().cpu().numpy())
            mse_val += loss(output2, batch_y).item()*batch_x.shape[0]
    preds2 = np.concatenate(preds2)
    true = np.concatenate(true)
    
    if min_val_loss > mse_val**0.5:
        min_val_loss = mse_val**0.5
        print("Saving...")
        torch.save(model2.state_dict(), "imv_tensor_lstm_nasdaq.pt")
        counter = 0
    else: 
        counter += 1
    
    if counter == patience:
        break
    print("Iter: ", i, "train: ", (mse_train/len(X_train_t1))**0.5, "val: ", (mse_val/len(X_val_t1))**0.5)
    if(i % 10 == 0):
        preds2 = preds2*(target_train_max2 - target_train_min2) + target_train_min2
        true = true*(target_train_max2 - target_train_min2) + target_train_min2
        mse = mean_squared_error(true, preds2)
        mae = mean_absolute_error(true, preds2)
        print("lr: ", opt2.param_groups[0]["lr"])
        print("mse: ", mse, "mae: ", mae)
        pl.figure(figsize=(20, 10))
        plt.plot(preds2)
        plt.plot(true)
        plt.show()

with torch.no_grad():
    mse_val = 0
    preds2 = []
    true2 = []
    alphas = []
    betas = []
    for batch_x, batch_y in data_test_loader1:
        batch_x = batch_x.cpu()
        batch_y = batch_y.cpu()
        output, a, b = model2(batch_x)
        output = output.squeeze(2)
        preds2.append(output.detach().cpu().numpy())
        true2.append(batch_y.detach().cpu().numpy())
        alphas.append(a.detach().cpu().numpy())
        betas.append(b.detach().cpu().numpy())
        mse_val += loss(output, batch_y).item()*batch_x.shape[0]
preds2 = np.concatenate(preds2)
true2 = np.concatenate(true2)

preds2 = preds2*(target_train_max2 - target_train_min2) + target_train_min2
true2 = true2*(target_train_max2 - target_train_min2) + target_train_min2

mse = mean_squared_error(true2, preds2)
mae = mean_absolute_error(true2, preds2)

pl.figure(figsize=(20, 10))
plt.plot(preds1)
plt.plot(true1)
plt.show()


alphas = np.concatenate(alphas)
betas = np.concatenate(betas)

alphas = alphas.mean(axis=0)
betas = betas.mean(axis=0)

alphas = alphas[..., 0]
betas = betas[..., 0]

alphas = alphas.transpose(1, 0)

fig, ax = plt.subplots(figsize=(10, 10))
im = ax.imshow(alphas)
ax.set_xticks(np.arange(X_train_t2.shape[1]))
ax.set_yticks(np.arange(len(data2.columns)))
ax.set_xticklabels(["t-"+str(i) for i in np.arange(X_train_t2.shape[1], 1, 1)])
ax.set_yticklabels(list(data2.columns))
for i in range(len(data1.columns)):
    for j in range(X_train_t2.shape[1]):
        text = ax.text(j, i, round(alphas[i, j], 3),
                       ha="center", va="center", color="w")
ax.set_title("Importance of features and timesteps for LITECOIN")
#fig.tight_layout()
plt.show()


plt.figure(figsize=(10, 10))
plt.title("Feature importance for LITECOIN")
plt.bar(range(len(data2.columns)), betas)
plt.xticks(ticks=range(len(data2.columns)), labels=list(data2.columns), rotation=90)
