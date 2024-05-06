import math
import numpy as np
from matplotlib import pyplot
import matplotlib.pyplot as plt 
 
# rectified linear function
def rectified(x):
    return max(0.0, x)
 # elu function
def elu(x,alpha):
    if x>=0:
       x_out=x 
    else: 
       x_out=alpha*(math.exp(x)-1)
    return x_out
 # leaky relu function
def leakrelu(x):
    return max(0.01*x, x)
 # tanh function
def tanh(x):
    x_out=(math.exp(x)-math.exp(-x))/(math.exp(x)+math.exp(-x))
    return x_out
 # sigmoid function
def sig(x):
 return 1/(1 + np.exp(-x))
 # shifted softplus function
def ssp(x):    
   return math.log(0.5*math.exp(x)+0.5)
 
# define a series of inputs
alpha=10
series_in = [x for x in range(-10, 11)]
# calculate outputs for our inputs
series_out = [rectified(x) for x in series_in]
series_out_elu = [elu(x,alpha) for x in series_in]
series_out_leakrelu = [leakrelu(x) for x in series_in]
series_out_tanh = [tanh(x) for x in series_in]
series_out_sig = [sig(x) for x in series_in]
series_out_ssp= [ssp(x) for x in series_in]

figure, axis = plt.subplots(ncols=2, nrows=2, figsize=(5.0, 4.5)
                   )
plt.suptitle('(Commonly used) Activation Functions', fontsize=10)
  
# For ReLu Function 
axis[0, 0].plot(series_in, series_out) 
axis[0, 0].set_title("ReLu", fontsize=10)
  
# For Tanh Function 
axis[0, 1].plot(series_in, series_out_tanh) 
axis[0, 1].set_title("Tanh", fontsize=10)
  
# For Sigmoid Function 
axis[1, 0].plot(series_in, series_out_sig) 
axis[1, 0].set_title("Sigmoid", fontsize=10)
  
# For Elu Function 
axis[1, 1].plot(series_in, series_out_elu) 
axis[1, 1].set_title("Elu", fontsize=10)
plt.tight_layout()
# Combine all the operations and display 
plt.show() 
