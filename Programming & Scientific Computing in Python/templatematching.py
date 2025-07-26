# -*- coding: utf-8 -*-
"""
Created on Mon May 30 00:10:38 2022

@author: weiwe
"""
import numpy as np
import pickle
from matplotlib import pyplot as plt
# load the database of labeled number images:
# original dataset: http://yann.lecun.com/exdb/mnist/

def show_templates(average_templates):
    n_templates = len(average_templates)
    fig, axs = plt.subplots(n_templates)
    for i in range(n_templates):
        axs[i].imshow(average_templates[i], cmap=plt.get_cmap('gray'))
    plt.show()
    return

file = open('C:\Python\MNIST.dat', 'rb')
MNIST = pickle.load(file)
file.close()
images = MNIST[0]
labels = MNIST[1]
shape_image = images[0].shape_image

'''
# show a single number image plus label:
plt.figure()
plt.imshow(images[0], cmap=plt.get_cmap('gray'))
plt.title(labels[0])
plt.show()
'''
m=0
average_template=np.zeros((9,28,28))
for i in range(len(labels)):
    if labels[i]==m:
        average_template[0,]
        

for j in range(10):
    show_templates(average_template[j])
    
