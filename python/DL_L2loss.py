# coding: utf-8

# ## Import the library

# In[69]:

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np
from numpy.random import randn


# ## layer definition (Need to do!!!)

# In[70]:


def InnerProduct_ForProp(x,W,b):    
    y = np.matmul(x, W) 
    row = y.shape[0]
    #add b for every row
    for i in range (row):
        y[i, :] + b
    return y

def InnerProduct_BackProp(dEdy,x,W,b):
    dEdx = np.matmul(dEdy, W.T) 
    dEdW = np.matmul(x.T, dEdy)
    #row = b.shape[1]
    #row = dEdy.shape[0]
    #array of 1 of b's size multiply dEdy
    size = dEdy.shape[0]
    dEdb = np.dot(np.ones((1, size)), dEdy)
    return dEdx,dEdW,dEdb 

def loss_ForProp(y,y_pred):
    #for 幾筆data
    loss = np.square(y_pred-y).sum()
    return loss

def L2Loss_BackProp(y,t):
    dEdx = y - t
    return dEdx
'''
def Sigmoid_ForProp(x):
    return 1 / (1 + np.exp(-x))


def Sigmoid_BackProp(dEdy,x):
    dEdx = dEdy*(1-x)*x
    return dEdx
'''
def ReLu_ForProp(x):
    return np.maximum(0,x)

def ReLu_BackProp(dEdy,x):
    x[x > 0] = 1
    x[x <= 0] = 0
    return (dEdy * x)

# ## Setup the Parameters and Variables (Can tune that!!!)

# In[71]:


eta =  0.00001      #learning rate
Data_num = 784      #size of input data   (inputlayer)

W1_num = 50         #size of first neural (1st hidden layer)
W2_num = 20
Out_num =  10      #size of output data  (output layer)

     

iteration =  1000   #epoch for training   (iteration)
image_num = 60000   #input images
test_num  = 10000   #testing images

## Cross Validation ##
##spilt the training data to 80% train and 20% valid##
train_num = int(image_num*0.8)  
valid_num = int(image_num*0.2)

# ## Setup the Data (Create weight array here!!!)

# In[72]:


w_1= (np.random.normal(0,1,Data_num*W1_num)).reshape(Data_num,W1_num)/100
w_2= (np.random.normal(0,1,W1_num*W2_num)).reshape(W1_num,W2_num)/100
w_out  = (np.random.normal(0,1,W2_num*Out_num)).reshape(W2_num, Out_num)/100
b_1= randn(1,W1_num)/100
b_2 = randn(1,W2_num)/100
b_out = randn(1,Out_num)/100
w_test_1= (np.random.normal(0,1,test_num*W1_num)).reshape(test_num,W1_num)/100
w_test_2= (np.random.normal(0,1,W1_num*W2_num)).reshape(W1_num,W2_num)/100
w_test_out  = (np.random.normal(0,1,W2_num*Out_num)).reshape(W2_num, Out_num)/100
b_test_1= randn(1,W1_num)/100
b_test_2 = randn(1,W2_num)/100
b_test_out = randn(1,Out_num)/100

print("w1 shape:", w_1.shape)
print("w_out shape:", w_out.shape)
print("b_1 shape:", b_1.shape)
print("b_out shape:", b_out.shape)
#print(w_out)


# ## Prepare all the data

# ### Load the training data and labels from files

# In[73]:


df = pd.read_csv('fashion-mnist_train_data.csv')
fmnist_train_images = df.as_matrix()
print("Training data:",fmnist_train_images.shape[0])
print("Training data shape:",fmnist_train_images.shape)

df = pd.read_csv('fashion-mnist_test_data.csv')
fmnist_test_images = df.as_matrix()
print("Testing data:",fmnist_test_images.shape[0])
print("Testing data shape:",fmnist_test_images.shape)

df = pd.read_csv('fashion-mnist_train_label.csv')
fmnist_train_label = df.as_matrix()
print("Training label shape:",fmnist_train_label.shape)


# ### Show the 100 testing images

# In[74]:
'''
plt.figure(figsize=(20,20))
for index in range(100):
    image = fmnist_test_images[index].reshape(28,28)
    plt.subplot(10,10,index+1,)
    plt.imshow(image)
plt.show() 
'''

# ### Convert the training labels data type to one hot type

# In[75]:


label_temp = np.zeros((image_num,10), dtype = np.float32)
for i in range(image_num):
    label_temp[i][fmnist_train_label[i][0]] = 1
train_labels_onehot = np.copy(label_temp)
print("Training label shape:",train_labels_onehot.shape)


# ### Separate train_images, train_labels into training and validating 

# In[76]:


train_data_img = np.copy(fmnist_train_images[:train_num,:])
train_data_lab = np.copy(train_labels_onehot[:train_num,:])
valid_data_img = np.copy(fmnist_train_images[train_num:,:])
valid_data_lab = np.copy(train_labels_onehot[train_num:,:])
# Normalize the input data between (0,1)
train_data_img = train_data_img/255.
valid_data_img = valid_data_img/255.
test_data_img = fmnist_test_images/255.

print("Train images shape:",train_data_img.shape)
print("Train labels shape:",train_data_lab.shape)
print("Valid images shape:",valid_data_img.shape)
print("Valid labels shape:",valid_data_lab.shape)
print("Test  images shape:",test_data_img.shape)


# ## Execute the Iteration (Need to do!!!)

# In[77]:


valid_accuracy = []
#train_accuracy = []
tacc = 0
vacc = 0

for i in range(iteration):
    # Forward-propagation
    y1 = InnerProduct_ForProp(train_data_img,w_1,b_1)
    y2 = ReLu_ForProp(y1)
    
    y1_2 = InnerProduct_ForProp(y2,w_2,b_2)
    y2_2 = ReLu_ForProp(y1_2)
    
    y3 = InnerProduct_ForProp(y2_2,w_out,b_out)
    
    loss = loss_ForProp(y3,train_data_lab)
    print("Train loss:",loss)
    
    # Bakcward-propagation
    loss_b = L2Loss_BackProp(y3,train_data_lab)
    
    dEdx_out_2, dEdw_out_2, dEdb_out_2 = InnerProduct_BackProp(loss_b,y2_2,w_out,b_out)
    dy = ReLu_BackProp(dEdx_out_2,y2_2)
    
    dEdx_out, dEdw_out, dEdb_out = InnerProduct_BackProp(dy,y2,w_2,b_2)
    dy = ReLu_BackProp(dEdx_out,y2)
    
    dEdx_1, dEdw_1, dEdb_1 = InnerProduct_BackProp(dy,train_data_img,w_1,b_1)
    
    # Parameters Updating (Gradient descent)
    w_1 = w_1 - eta * dEdw_1
    w_2 = w_2 - eta * dEdw_out
    w_out = w_out - eta * dEdw_out_2
    
    b_1 = b_1 - eta * dEdb_1
    b_2 = b_2 - eta * dEdb_out
    b_out = b_out - eta * dEdb_out_2
    
    '''
    # Forward-propagation
    y1 = InnerProduct_ForProp(train_data_img,w_1,b_1)
    y2 = ReLu_ForProp(y1)
    
    y1_2 = InnerProduct_ForProp(train_data_img,w_1,b_1)
    y2_2 = ReLu_ForProp(y1_2)
    
    y3 = InnerProduct_ForProp(y2_2,w_out,b_out)
    loss = loss_ForProp(y3,train_data_lab)
    print("Train loss:",loss)
    
    # Bakcward-propagation
    loss_b = L2Loss_BackProp(y3,train_data_lab)
    dEdx_out, dEdw_out, dEdb_out = InnerProduct_BackProp(loss_b,y2_2,w_out,b_out)
    
    dy = ReLu_BackProp(dEdx_out,y2_2) 
    dEdx_out_2, dEdw_out_2, dEdb_out_2 = InnerProduct_BackProp(loss_b,y2_2,w_out,b_out)
    
    dy_2 = ReLu_BackProp(dEdx_out,y2) 
    dEdx_out, dEdw_out, dEdb_out = InnerProduct_BackProp(dy_2,train_data_img,w_1,b_1)
    '''
    '''
    # Forward-propagation
    y1 = InnerProduct_ForProp(train_data_img,w_1,b_1)
    y2 = ReLu_ForProp(y1)
    y3 = InnerProduct_ForProp(y2,w_out,b_out)
    loss = loss_ForProp(y3,train_data_lab)
    print("Train loss:",loss)
    
    # Bakcward-propagation
    loss_b = L2Loss_BackProp(y3,train_data_lab)
    dEdx_out, dEdw_out, dEdb_out = InnerProduct_BackProp(loss_b,y2,w_out,b_out)
    dy = ReLu_BackProp(dEdx_out,y1) 
    dEdx_1, dEdw_1, dEdb_1 = InnerProduct_BackProp(dy,train_data_img,w_1,b_1)
    
    # Parameters Updating (Gradient descent)
    w_1 = w_1 - eta * dEdw_1
    w_out = w_out - eta * dEdw_out
    b_1 = b_1 - eta * dEdb_1
    b_out = b_out - eta * dEdb_out
    
    # Forward-propagation
    y1_2 = InnerProduct_ForProp(train_data_img,w_1,b_1)
    y2_2 = ReLu_ForProp(y1_2)
    y3_2 = InnerProduct_ForProp(y2_2,w_out,b_out)
    loss_2 = loss_ForProp(y3_2,train_data_lab)
    print("Train loss:",loss_2)
    
    # Bakcward-propagation
    loss_b_2 = L2Loss_BackProp(y3_2,train_data_lab)
    dEdx_out_2, dEdw_out_2, dEdb_out_2 = InnerProduct_BackProp(loss_b_2,y2_2,w_out,b_out)
    dy_2 = ReLu_BackProp(dEdx_out_2,y1_2) 
    dEdx_1_2, dEdw_1_2, dEdb_1_2 = InnerProduct_BackProp(dy_2,train_data_img,w_1,b_1)
    '''
    
    
  
    # Do cross-validation to evaluate model
    #換成test data#百分比
    valid_y1 = InnerProduct_ForProp(valid_data_img,w_1,b_1)
    valid_y2 = ReLu_ForProp(valid_y1)
    
    valid_y1_2 = InnerProduct_ForProp(valid_y2,w_2,b_2)
    valid_y2_2 = ReLu_ForProp(valid_y1_2)
    
    valid_y3 = InnerProduct_ForProp(valid_y2_2,w_out,b_out)
    
    valid_loss = loss_ForProp(valid_y3,valid_data_lab)
    print("valid loss:",valid_loss)

    # Get 1-D Prediction array
    # Compare the Prediction and validation
    for i in range (train_num):
        #print("train_data_index:",train_data_lab)
        train_data_index = np.argmax(train_data_lab[i,:])
        train_pred_index = np.argmax(y3[i,:])
        if (train_data_index == train_pred_index):
            tacc += 1
            
    for i in range (valid_num):       
        #print("valid_data_lab:",valid_data_lab[i,:])
        valid_data_index = np.argmax(valid_data_lab[i,:])
        
        valid_pred_index = np.argmax(valid_y3[i,:])
        if (valid_data_index == valid_pred_index):
            vacc += 1
    vacc = vacc/valid_num
    valid_accuracy.append(vacc)
    if (i+1)%10 == 0:
        print(vacc)
            
        
    #print('valid_acc:',v_acc/12000)
    #Calculate the accuracy


# ## Testing Stage

# ### Predict the test images (Do forward propagation again!!!)

# In[81]:


# Forward-propagation
#test_y1 = InnerProduct_ForProp(test_data_img,w_1,b_1)
#test_y2 = ReLu_ForProp(test_y1)
#test_y3 = InnerProduct_ForProp(test_y2,w_out,b_out)
#test_Out_data = InnerProduct_ForProp(test_y2,w_out,b_out)
#test_loss = loss_ForProp(test_Out_data,test_data_lab)

#test_y1 = InnerProduct_ForProp(test_data_img,w_test_1,b_test_1)
#test_y2 = ReLu_ForProp(test_y1)
    
#test_y1_2 = InnerProduct_ForProp(test_y2,w_test_2,b_test_2)
#test_y2_2 = ReLu_ForProp(test_y1_2)
    
#test_y3 = InnerProduct_ForProp(test_y2_2,w_test_out,b_test_out)



# ### Convert results to csv file (Input the (10000,10) result array!!!)

# In[82]:

'''
# Convert "test_Out_data" (shape: 10000,10) to "test_Prediction" (shape: 10000,1)
test_Prediction = np.argmax(test_Out_data, axis=1)[:,np.newaxis].reshape(test_num,1)
df = pd.DataFrame(test_Prediction,columns=["Prediction"])
df.to_csv("DL_LAB1_prediction_ID.csv",index=True, index_label="index")
'''

# ## Convert results to csv file

# In[83]:


accuracy = np.array(valid_accuracy)
plt.plot(accuracy, label="$iter-accuracy$")
y_ticks = np.linspace(0, 100, 11)
plt.legend(loc='best')
plt.xlabel('iteration')
plt.axis([0, iteration, 0, 100])
plt.ylabel('accuracy')
plt.show()

