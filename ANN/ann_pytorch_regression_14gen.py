# -*- coding: utf-8 -*-
"""
Created on Mon Sep 30 07:53:03 2019

@author: drshr
"""

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

import ga_ceed_14gen

data_14_gen_df = pd.read_excel('ceed_data_14_gen.xlsx')
#x_data = Variable(torch.Tensor([[1.0], [2.0], [3.0]])) 
#y_data = Variable(torch.Tensor([[2.0,3.0], [4.0,6.0], [6.0,9.0]])) 

#data = pd.read_excel("data.xlsx")
data = pd.read_excel("ga_data_4_ann_14gen.xlsx")
x_data=data[data.columns[0:1]]
x_data = Variable(torch.Tensor(x_data.values))

y_data=data[data.columns[1:-2]]
y_data = Variable(torch.Tensor(y_data.values))

no_of_itr =1000
best_res = np.zeros([no_of_itr,2])

class LinearRegressionModel(torch.nn.Module): 
  
    def __init__(self): 
        super(LinearRegressionModel, self).__init__() 
        self.linear = torch.nn.Linear(1,13)  # One in and one out 
  
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

result_list = np.zeros([80, 16])  
j =0
for pdemand in range(1400, 3000, 20):
    new_var = Variable(torch.Tensor([[pdemand]])) 
    pred_y = our_model(new_var).data
    result_list[j][0] = pdemand
    print(pred_y)
    pred_y_arr =  pred_y.detach().numpy()
    print(pred_y_arr[0])
    result_list[j][1:14] = pred_y_arr
    result_list[j][14] = pdemand - sum(pred_y_arr[0])    
    result_list[j][15] = ga_ceed_14gen.calc_candidate_total_cost(result_list[j][1:15], data_14_gen_df, pdemand)
    print("predict (after training)", our_model(new_var).data) 
    j =j +1

#init_loss = 100000
#best_loss = min(best_res[:][1])

#plt.plot(best_res)
#plt.ylim(best_loss, init_loss)
#plt.xlabel("Iteration")
#plt.ylabel("Loss")
#plt.show()    
    

best_solution_df = pd.DataFrame(result_list)
best_solution_df.to_excel('data_4m_ann_ga_14.xlsx')    