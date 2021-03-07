import torch 
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt
import load_data
import CNN
import train
import test
import Visualize_conf_matrix
import Data_visualization
import retrieval
import evaluation
import feature_extraction
import KNN_evaluation

# Device configuration
device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

# Hyper parameters
num_epochs = 5
num_classes = 10
batch_size = 100
learning_rate = 0.001

#Data Visualization
Data_visualization.data_visualize(load_data.train_loader)

#Model define
model = CNN.CNN(num_classes).to(device)

# Loss and optimizer
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=learning_rate)

# Train the model
train.train_model(num_epochs,model,criterion,optimizer, load_data.train_loader)

# Test the model
pred,labelss = test.test_model(model,load_data.test_loader,device)

# Save the model checkpoint
torch.save(model.state_dict(), 'modelSVHN.ckpt')

#Visualize confusion matrix
Visualize_conf_matrix.confusion_matrix_plot(labelss, pred)

#Visualize histogram for class unbalanced

plt.hist(labelss, bins=10)
plt.show()
"""We can observe an unbalance in the dataset since all the clases have not the same number of examples. It could be solved with data augmentation for the classes that appear less often."""

#%%
#Retrieval
'''
#Find all images that are the same as a given one. One way of doing so is
#by computing a feature vector representation of the image through a neural net and 
#getting the closest images. If the net is already trained the feature representation will
#be similar for similar images and different for differents images.
'''

#Get features of test and train set

features_test = feature_extraction.features_test_extraction(load_data.test_loader,model)

features_train, labels_train_total= feature_extraction.features_train_extraction(load_data.train_loader,model)

#Lets check whether this method is accurate enough or not.
#If yes our posterior retrieval will be satisfactory
#It classifies with 0.84 accuracy, which is good enough

KNN_accuracy = KNN_evaluation.KNN_eval(features_train, features_test, labels_train_total, labelss)

#%%
#Now lets do the retrieval process. Given a string or a sample we have to 
#retrieve all the samples that belong to the same class.
#Here as the dataset is not big we can have all the data in memory. However,
#if it wouldnt be the case, we could retrive by batches.

#First we evaluate QbE. So given an image retrieve the k top nearest images that belong to the
#same class.

#Select image: ( change this number)

imagen_query=7
k=len(features_test) #Select how many images you want to retrieve. You get the k best results
retrieval_list_QbE  = retrieval.QbE_k_items(imagen_query,k,features_test,labelss)

#Select index (QbS): ( change this number)
index_search=5
retrieval_list_QbS  = retrieval.QbS_k_items(index_search,k,features_test,labelss)

#%%
#Evaluation mAP. A number of samples is selected
mAP_accuracy = evaluation.mAP(features_test,labelss,x=100)   
print(mAP_accuracy)

#Result for x = 100. mAP = 0.57707 . (So, the mean of the AP of 100 samples)
#%%
image_query=7
print(evaluation.AP(image_query,features_test,labelss))






