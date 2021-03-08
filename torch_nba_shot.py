#BasketBall Shot model
import pandas as pd
import torch
from torch import nn, optim
from torch.utils.data import DataLoader

import matplotlib.pyplot as plt

#df = pd.read_csv('C:\\Users\\user\\Documents\\Projects\\Sampe_Data\\NBA_shotdata\\shot_logs\\shot_logs_all.csv')

class Data():
    def __init__(self,train):
              

        if (train):
            data = pd.read_csv('C:\\Users\\user\\Documents\\Projects\\Sampe_Data\\NBA_shotdata\\shot_logs\\shot_logs_all_ii.csv')
            df_size = len(data)
            #print(df_size)
            train_length=df_size*80//100
            print(train_length)
            self.x= torch.tensor(data.iloc[0:train_length,:].drop(['SHOT_RESULTS'],axis=1).values, dtype=torch.float) #n 7090 is the number of samples and '' is the y column label
            self.x=self.x/torch.mean(self.x)
            self.y=torch.tensor(data.loc[0:train_length,'SHOT_RESULTS'], dtype=torch.float).reshape((train_length+1,1)) # n should be about 80% of the data
            self.len=self.x.shape[0]
        else:
            data = pd.read_csv('C:\\Users\\user\\Documents\\Projects\\Sampe_Data\\NBA_shotdata\\shot_logs\\shot_logs_all_ii_test.csv')
            df_size = len(data)
            #print(df_size)
            train_length=df_size
            print(train_length)
            self.x= torch.tensor(data.iloc[0:train_length,:].drop(['SHOT_RESULTS'],axis=1).values, dtype=torch.float) #n 7090 is the number of samples and '' is the y column label
            self.x=self.x/torch.mean(self.x)
            self.y=torch.tensor(data.loc[0:train_length,'SHOT_RESULTS'], dtype=torch.float).reshape((train_length,1)) # n should be about 80% of the data
            self.len=self.x.shape[0]
    def __getitem__(self,index):
        return self.x[index], self.y[index]

    def __len__(self):
        return self.len

shot_dataset_train = Data(train=True)
print("Training Data set Created")
shot_dataset_test = Data(train=False)
batches=750 #750
learnin_rate = 0.004 #0.004
epochs = 10 #10

train_loader = DataLoader(dataset=shot_dataset_train, batch_size=batches)
test_loader = DataLoader(dataset=shot_dataset_test, batch_size=batches)

class logistic_reg(nn.Module):
    def __init__(self,inputs,outputs):
        super(logistic_reg, self).__init__()
        self.linear = nn.Linear(inputs,outputs)

    def forward(self,x):
        out = torch.sigmoid(self.linear(x))
        return out

print("NN Model Initialised")

model = logistic_reg(4,1)

print("Model Parameters set")

optimizer = optim.Adam(model.parameters(), lr = learnin_rate) #0.02-0.005
print("Adams Optimizer is set, with a learning rate: ", learnin_rate)
criterion = nn.BCELoss()
print("BCE Loss Initialised")

print("Epochs: ", epochs)

LOSS = []
Accuracy = []
for epoch in range(epochs):
    for x,y in train_loader:
        yhat = model(x)
        loss = criterion(yhat,y)
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        LOSS.append(loss.item())

    correct =0
    model.eval()
    for x_test, y_test in test_loader:
        z = model(x_test)
        z=torch.round(z)
        correct += (z == y_test).sum().item()

    Accuracy.append(correct/len(shot_dataset_test))


    


#plot the loss
headarg_1 = 'NBA Shot predictor (LR/BS/E = '
headarg_2 = str(learnin_rate)
headarg_3 = str(batches)
headarg_4 = str(epochs)

color = 'tab:red'
plt.plot(LOSS,color=color)
plt.xlabel('Epoch',color=color)
plt.ylabel('Cost',color=color)
plt.title(headarg_1 + headarg_2+ '/' + headarg_3 + '/' +headarg_4+')')
plt.show()

color = 'tab:blue'
plt.plot(Accuracy,color=color)
plt.xlabel('Epoch',color=color)
plt.ylabel('Accuaery',color=color)
plt.title('Accuracy of the model')
plt.show()