# -*- coding: utf-8 -*-
"""
Created on Tue Aug  3 22:37:22 2021

@author: Chaobang Huang and Yixiu Zhu
"""

import numpy as np
import tensorflow as tf
import re
import matplotlib.pyplot as plt
from sklearn.utils.class_weight import compute_class_weight
from scipy import spatial
from sklearn.model_selection import train_test_split

Batch_size=1024
Epoch=2 # 50
path="text1.txt"
WORD_LENGTH = 5
rate=0.01 # 0.01




"""read and store data as long string"""
file = open(path,'r',encoding='UTF-8')
text=file.read().lower()
file.close()

"""get rid of special characters and split the long string into list of string by space"""
text=re.sub("[^\w\s]", "", text) # get rid of punctuations
words=text.split()               # get list of strings

"""create a dictionary with unique string as the key"""

def eliminate(list):  # return the index where 'a' first appears in the sorted list of strings
    # list -> int
    k=0
    for i in list:
        k+=1
        if i[0]=='a':
            return k-1
    return k-1

unique_words = np.unique(words) 
unique_words=unique_words[eliminate(unique_words):]



    
"""analyze distribution of classes"""
distribution={}
for i, c in enumerate(words):
    if c in unique_words:
        if c not in distribution:
            distribution[c]=1
        else:
            distribution[c]+=1
        
def counter(i): # i is number of times a word qppears in the data, it returns number of qualified words
    count=0
    for x in distribution:
        if distribution[x]==i:
            count+=1
    return count

x_axis=np.arange(len(distribution))
y_axis=np.zeros_like(x_axis) 
q=0
for i in distribution:
    y_axis[q]=distribution[i]
    q+=1
    

plt.scatter(x_axis,y_axis,marker="x",s=2.5)
plt.xlabel("index")
plt.ylabel("occurences")
plt.show() # do it to clear the current picture in buffer and let it appear immediately





def loadGloveModel(File):
    print("loading the text file")
    f = open(File,'r',encoding='UTF-8')
    gloveModel = {}
    for line in f:
        splitLines = line.split()
        word = splitLines[0]
        word_vector = np.array([float(value) for value in splitLines[1:]])
        gloveModel[word] = word_vector
    print(len(gloveModel),"successfully loaded!")
    return gloveModel

glovelModel=loadGloveModel("glove.6B.50d.txt") # hashtable (dictionary)
glovelModel["<unknown>"]=np.zeros(50)          # account for words not in the hashtable










"""get raw features(list of list of strings) and corresponding labels(list of words)"""
prev_words = [] # the list of list consisting of "WORDS_LENGTH" words
next_words = [] # list of words that are next words
for i in range(len(words) - WORD_LENGTH):
    prev_words.append(words[i:i + WORD_LENGTH])
    next_words.append(words[i + WORD_LENGTH])
    
    
""" feature engineering """    
X = np.zeros((len(prev_words), WORD_LENGTH, 50))
Y = np.zeros((len(next_words), 50))
   
for i in range(len(prev_words)):
    for j in range(len(prev_words[i])):
        if prev_words[i][j] in glovelModel:
            X[i,j] = glovelModel[prev_words[i][j]] 
        else:
            X[i,j]=glovelModel["<unknown>"] 
    if next_words[i] in glovelModel:        
        Y[i] = glovelModel[next_words[i]]
    else:
        Y[i]=glovelModel["<unknown>"]

count=0        
for i in next_words:
    if i not in glovelModel:
        count+=1

# Note Although we split the data here, we just used model.fit's API to do the separation during training
# since we will test the model on all of the original sentences instead of just testing data
x_train, x_valid, y_train, y_valid = train_test_split(X, Y, test_size=0.01, shuffle= True)    
 
# =============================================================================
#     
#  
# """feature engineering for features and labels by one-hot encoding"""
# X = np.zeros((len(prev_words), WORD_LENGTH, len(unique_words)), dtype=np.ubyte)
# Y = np.zeros((len(next_words), len(unique_words)), dtype=np.ubyte)
# 
# for i in range(len(prev_words)):
#     for j in range(len(prev_words[i])):
#         X[i, j, unique_word_index[prev_words[i][j]]] = 1
#     Y[i, unique_word_index[next_words[i]]] = 1
#     
#     
# # =============================================================================
# # alternative way of doing the same feature engineering
# # for i, each_words in enumerate(prev_words):
# #     for j, each_word in enumerate(each_words):
# #         X[i, j, unique_word_index[each_word]] = 1
# #     Y[i, unique_word_index[next_words[i]]] = 1
# # =============================================================================
#      
#   
# =============================================================================

 

# RNN with LSTM:

"""build"""
model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(128, return_sequences=True,input_shape=(WORD_LENGTH, 50))) # one example: (1, 5, 50), and the batch size (first dimension) would be taken care by the API
model.add(tf.keras.layers.Dropout(0.5)) 
model.add(tf.keras.layers.LSTM(128))
model.add(tf.keras.layers.Dense(50)) 
#model.add(tf.keras.layers.Activation('softmax'))  # Do not use softmax since the label is not a probability vector!

"""train"""



def euclidean_distance_loss(y_true, y_pred):
    # return tensor
    return tf.keras.backend.sqrt(tf.keras.backend.sum(tf.keras.backend.square(y_pred - y_true), axis=-1))

def find_closest_embeddings(representation):
    # return a long list of string
    return sorted(glovelModel.keys(), key=lambda word: spatial.distance.euclidean(glovelModel[word], representation))



steps=5*len(X)/Batch_size

lr_schedule = tf.keras.optimizers.schedules.ExponentialDecay(
    initial_learning_rate=rate,
    decay_steps=steps,
    decay_rate=0.96,
    staircase=True) 

opt = tf.keras.optimizers.Adam(learning_rate=lr_schedule)

model.compile(loss=euclidean_distance_loss, 
              optimizer=opt,metrics=["cosine_similarity"])

stop=tf.keras.callbacks.EarlyStopping(monitor='cosine_similarity',mode="max",patience=20,
                                      restore_best_weights=True)

checkpoint = tf.keras.callbacks.ModelCheckpoint(
    filepath="checkpoint_words",
    save_weights_only=True,
    monitor="cosine_similarity",
    mode='max',
    save_best_only=True)

#model.load_weights("checkpoint_words")
#model.save_weights("checkpoint_words")       

    
history = model.fit(x=X,y=Y,validation_split=0.001,
                    batch_size=Batch_size, epochs=Epoch, callbacks=[stop,checkpoint],
                    shuffle=True)
 





"""draw"""
plt.plot(history.history['cosine_similarity'])
plt.plot(history.history['val_cosine_similarity'])
plt.title('model similarity')
plt.ylabel('cosine similarity')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()

plt.plot(history.history['loss'])
plt.plot(history.history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train', 'test'], loc='upper left')
plt.show()


"""testing"""

# convert a 5-word sentence to representation of glove vectors in dimension of (1,5,50)
def create_input(sentence): 
    output=np.zeros(shape=(1,5,50))
    sentence=sentence.lower().split() 
    for i in range(len(sentence)):
        if sentence[i] in glovelModel:
            output[0,i]=glovelModel[sentence[i]]
        else:
            output[0,i]=glovelModel["<unknown>"]
    return output


def get_next5(text):
    x=create_input(text.lower())
    y_hat=model.predict(x)
    new_word=find_closest_embeddings(y_hat)[0:5]
    return new_word

def get_next(text):
    x=create_input(text.lower())
    y_hat=model.predict(x)
    new_word=find_closest_embeddings(y_hat)[0]
    return new_word

def generator(text,number_words):
    if number_words==1:
        temp=text.split()
        temp=temp[-1:-(WORD_LENGTH+1):-1][-1::-1]
        input=""
        for i in range(len(temp)):
            input+=(" "+ temp[i])
        x=get_next(input)
        #print(input)
        output=text+" "+x
        return output
    else:
        temp=text.split()
        temp=temp[-1:-(WORD_LENGTH+1):-1][-1::-1]
        input=""
        for i in range(len(temp)):
            input+=(" "+ temp[i])
        x=get_next(input)
        output=generator(text+" "+x,number_words-1)
        return output


Random_number=1000
for i in range(9):
    x=X[Random_number+i] #(5,50)
    x=np.reshape(x,(1,5,50))
    y_hat=model.predict(x)
    print("predictions are: ", find_closest_embeddings(y_hat)[0:5])
    print("true label is: ", find_closest_embeddings(Y[Random_number+i])[0])
    print("")
    


"""demonstration of coscine similarity and loss function"""

a=glovelModel["person"] 
temp=find_closest_embeddings(a)[0:5] 
print("top 5 five words:", temp)      
b=glovelModel[temp[0]]
c=glovelModel[temp[1]]
d=glovelModel[temp[2]]
print("")
print("similarity between a and b:",a@b/(np.linalg.norm(a)*np.linalg.norm(b)))
print("distance between a and b:",euclidean_distance_loss(a,b).numpy())
print("")
print("similarity between a and c:",a@c/(np.linalg.norm(a)*np.linalg.norm(c)))
print("distance between a and c:",euclidean_distance_loss(a,c).numpy())
print("")
print("similarity between a and d:",a@d/(np.linalg.norm(a)*np.linalg.norm(d)))
print("distance between a and d:",euclidean_distance_loss(a,d).numpy())






# =============================================================================
# 
# """GRU"""
# model_GRU = tf.keras.models.Sequential()
# model_GRU.add(tf.keras.layers.GRU(128, return_sequences=True,input_shape=(WORD_LENGTH, 50)))
# # one example: 5, 50, and the batch would be taken care by the API
# model_GRU.add(tf.keras.layers.GRU(128))
# # =============================================================================
# # model.add(tf.keras.layers.Dense(128))
# # model.add(tf.keras.layers.Dropout(0.5)) 
# # =============================================================================
# model_GRU.add(tf.keras.layers.Dense(50)) 
# 
# opt = tf.keras.optimizers.Adam(learning_rate=0.01)
# 
# model_GRU.compile(loss=euclidean_distance_loss, 
#               optimizer=opt,metrics=["cosine_similarity"])
# # =============================================================================
# # stop=tf.keras.callbacks.EarlyStopping(monitor='val_categorical_accuracy',patience=20,
# #                                       restore_best_weights=True)
# # 
# # checkpoint = tf.keras.callbacks.ModelCheckpoint(
# #     filepath="checkpoint_words",
# #     save_weights_only=True,
# #     monitor='val_categorical_accuracy',
# #     mode='max',
# #     save_best_only=True)
# # =============================================================================
# 
# #model.load_weights("checkpoint_words")
#                    
# history_GRU = model_GRU.fit(X, Y, validation_split=0.001,batch_size=1024, epochs=100, shuffle=True)
# 
# =============================================================================
