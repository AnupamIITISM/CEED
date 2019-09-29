# -*- coding: utf-8 -*-
"""
Created on Sun Sep 29 10:02:11 2019

@author: drshr
"""
import pandas as pd
import numpy as np
import torch 
from torch.autograd import Variable 
import matplotlib.pyplot as plt


#x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]])) 
#y_data = Variable(torch.Tensor([[2.0,3.0], [4.0,6.0], [6.0,9.0]])) 

data = pd.read_excel("data.xlsx")
x_data=data[data.columns[0:1]]
x_data = Variable(torch.Tensor(x_data.values))

y_data=data[data.columns[1:-2]]
y_data = Variable(torch.Tensor(y_data.values))

no_of_itr =10000
best_res = np.zeros([no_of_itr,2])

class LinearRegressionModel(torch.nn.Module): 
  
    def __init__(self): 
        super(LinearRegressionModel, self).__init__() 
        self.linear = torch.nn.Linear(1,5)  # One in and one out 
  
    def forward(self, x): 
        y_pred = self.linear(x) 
        return y_pred   
    
# our model 
our_model = LinearRegressionModel() 
  
criterion = torch.nn.MSELoss(size_average = False) 
optimizer = torch.optim.Adam(our_model.parameters(), lr = 0.05) 
  
for epoch in range(no_of_itr): 
  
    # Forward pass: Compute predicted y by passing  
    # x to the model 
    pred_y = our_model(x_data) 
  
    # Compute and print loss 
    loss = criterion(pred_y, y_data) 
  
    # Zero gradients, perform a backward pass,  
    # and update the weights. 
    optimizer.zero_grad() 
    loss.backward() 
    optimizer.step() 
    print('epoch {}, loss {}'.format(epoch, loss.item()))
    best_res[epoch][0] = epoch
    best_res[epoch][1] = loss.item()
  
new_var = Variable(torch.Tensor([[750.0]])) 
pred_y = our_model(new_var) 
print("predict (after training)", 4, our_model(new_var).data) 

init_loss = 10000
best_loss = min(best_res[:][1])

plt.plot(best_res)
plt.ylim(best_loss, init_loss)
plt.xlabel("Iteration")
plt.ylabel("Loss")
plt.show()    