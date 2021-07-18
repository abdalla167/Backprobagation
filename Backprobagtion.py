import matplotlib.pyplot as plt
from tkinter import *
from tkinter import ttk
from tkinter import messagebox
from sklearn.utils import shuffle


import numpy as np
import random
import math
import sys
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import datasets, __all__, preprocessing
from sklearn.datasets import load_iris
from sklearn.utils import shuffle

###gloabel variabel


global list_all_number_each_hiden_layer
global list_all_matrix
list_all_number_each_hiden_layer=[]
list_all_matrix=[]

def loadFile(df):

    resultList = []
    f = open(df, 'r')
    for line in f:
        line = line.rstrip('\n')
        sVals = line.split(',')
       # fVals = list(map(np.str, sVals))
        resultList.append(sVals)
    f.close()
    data= np.asarray(resultList)  # not necessary
    df1 = pd.DataFrame(data)
    X1 = df1[0]
    X2 =  df1[1]
    X3 =  df1[2]
    X4 = df1[3]
    Classes=df1[4]
    Classes=np.array(Classes)
    total_class=["Iris-setosa","Iris-versicolor","Iris-virginica"]
    mapping = {}
    for x in range(len(total_class)):
        mapping[total_class[x]] = x

    one_hot_encode = []

    for c in Classes:
        arr = list(np.zeros(len(total_class), dtype=int))
        arr[mapping[c]] = 1
        one_hot_encode.append(arr)
    df1[4]=one_hot_encode

    return  df1


def train_test(datafram):

    datafram_copy = datafram.copy()
    datafram_copy = shuffle(datafram_copy)
    train_set = datafram_copy.sample(frac=0.60, random_state=1)
    test_set = datafram_copy.drop(train_set.index)
    return train_set,test_set

##get number of layer
def number_nurans(numbers_all):
    numbers_all = numbers_all.split(',')
    for i in range(0, len(numbers_all)):
        numbers_all[i] = int(numbers_all[i])
    list_all_number_each_hiden_layer=numbers_all
    return list_all_number_each_hiden_layer
def create_all_matrix (List_size_hiden_layer):
    #retrun from function list of matrix have all matrix but it is empty marix not have value
    list_all_matrix1=[]
    for i , value in enumerate(List_size_hiden_layer):
        if i==0:
            new_matrix=np.empty([4,value],dtype=float)
            list_all_matrix1.append(new_matrix)
        if i==len( List_size_hiden_layer)-1:
            new_matrix = np.empty([value,3], dtype=float)
            list_all_matrix1.append(new_matrix)
        else:
            new_matrix=np.empty([value,List_size_hiden_layer[i+1]],dtype=float)
            list_all_matrix1.append(new_matrix)
    list_all_matrix=list_all_matrix1

    return  list_all_matrix


# def intilaiz_random_wight(list_of_matrix,bias):
#     for i, valu in enumerate(list_of_matrix):
#         rows = len(valu)
#         columns = len(valu[0])
#         wight= np.random.randn( rows,columns )
#         list_of_matrix[i]=wight
#         if i< len(list_all_number_each_hiden_layer):
#          mat = np.random.randn(rows, list_all_number_each_hiden_layer[i])
#          if bias==0:
#            B.append(np.zeros(mat.shape[1]))
#          else:
#            B.append(np.random.randn(mat.shape[1]))
#     return list_of_matrix
#
###feed forward
def initializeW(neurons,bias,y):
        lis_mat_random_wight=[]
        B=[]
        prv = 4  # number of features in input
        # need number of hidden layers + 1 matrix for W
        # each of them has size of (number of current layer nodes * number of next layer nodes )
        # weight matrices between input layer and hidden layer and between all hidden layers

        for i in range(len(neurons)):
            mat = np.random.randn(prv, neurons[i])
            prv = neurons[i]
            lis_mat_random_wight.append(mat)
            if bias == 1:
               B.append(np.random.randn(mat.shape[1]))
            else:
               B.append(np.zeros(mat.shape[1]))
        # weight matrix between last hidden layer and output
        mat = np.random.randn(prv, len(y[0]))
        lis_mat_random_wight.append(mat)
        if bias == 1:
            B.append(np.random.randn(mat.shape[1]))
        else:
            B.append(np.zeros(mat.shape[1]))
        return B,lis_mat_random_wight
def activation( s,type):
        if type == "Sigmoid":
            return 1 / (1 + np.exp(-s))
        else:
            return np.tanh(s)
def feed_forwared(X,l_W_mat,B,type):
    a = list()
    y_predit=list()
    a.append(X)
    for i in range(len( list_all_number_each_hiden_layer)+1):
        z = np.dot(a[i], l_W_mat[i])+B[i]
        #use activation function to add non linarty and limet domain
        z = activation(z,type)
        a.append(z)
    y_predit.append(a[len(a) - 1])
    return a,y_predit,list_all_number_each_hiden_layer
def derivative( Error,activationFunction):
        if activationFunction != "Sigmoid":
          return 1 - Error ** 2
        else:
          return Error * (1 - Error)
def backword_probagation(out_put_from_feed_forwared,label_now,X_now,y_predict,bias ,learing_rate,list_h,l_w_m,B,type):


    final_prediction = y_predict
    layers_delta = list()
    layers_number=len(list_h )+1
    # cal gradint
    # (target - y) * derv(activation function)

    #for output layer only have target
    Error = label_now - out_put_from_feed_forwared[layers_number]
    delta_error = Error * derivative(out_put_from_feed_forwared[layers_number],type)
    layers_delta.append(delta_error)
    prv = delta_error
    #################
    ##gradint backword
    for i in reversed(range(layers_number-1)):
            Error = np.dot(l_w_m[i + 1], prv)
            Error_delta = Error * derivative(out_put_from_feed_forwared[i + 1],type)
            prv = Error_delta
            layers_delta.append(Error_delta)
    tem_wight=l_w_m
    ###for baias and update wight
    for i in range(0,layers_number):
        if bias == 1:
            B[i] += layers_delta[len(layers_delta) - i - 1] * learing_rate
        w=layers_delta[len(layers_delta) - i - 1] * np.atleast_2d(out_put_from_feed_forwared[i]).T * learing_rate
        l_w_m[i] += w
    return   l_w_m,B
def test(x,l_W_mat,B,type):
     a,y_p, list_h = feed_forwared(x,l_W_mat,B,type)
     return  a[len(a) - 1]
def train( X, y,list_all_number_each_hiden_layer,epoch,learing_rate,bias,type):
    y = np.array(y)
    y_predit=list()
    #X, neurons, bias, y

    # list_all_matrix = create_all_matrix(list_all_number_each_hiden_layer)
    # lis_mat_random_wight = intilaiz_random_wight(list_all_matrix, 1)
    B,lis_mat_random_wight  =initializeW(list_all_number_each_hiden_layer,bias,y)
    # a, y_p, list_h = feed_forwared([1, 5.1, 3.5, 1.4, 0.2])
    # backword_probagation(a, [1, 0, 0], [5.1, 3.5, 1.4, 0.2], y_p, 1, 0.001, list_h)
    for i in range(epoch):
        y_predit.clear()
        for j in range(len(X)):
            new_x=np.array( X.iloc[j],dtype=float)
            list_temp=[]
            #list_temp.append(1)
            list_temp.append(new_x[0])
            list_temp.append(new_x[1])
            list_temp.append(new_x[2])
            list_temp.append(new_x[3])
            list_temp1=np.array(list_temp)
            a,y_p, list_h = feed_forwared(list_temp1,lis_mat_random_wight,B,type)
            y_p1=y_p
            lis_mat_random_wight1,B1= backword_probagation(a, y[j],list_temp1,y_p1,bias,learing_rate,list_h,lis_mat_random_wight,B,type)
            lis_mat_random_wight=lis_mat_random_wight1
            B=B1
    return lis_mat_random_wight1,B1

def Run(list_layer,epoch,bias,learing_rate,type):
    global list_all_number_each_hiden_layer
    list_all_number_each_hiden_layer = number_nurans(list_layer)
    train_data,test_data= train_test(loadFile('IrisData.txt'))
    y_actioal=train_data[4]
    y_test=test_data[4]
    train_data.drop(train_data.columns[4], axis=1,inplace=True)
    test_data.drop(test_data.columns[4], axis=1,inplace=True)
    l_W_mat,B=train(train_data,y_actioal,list_all_number_each_hiden_layer,epoch,learing_rate,bias,type)
    y_predict_test=[]
    for i in range(len( test_data)):
        new_x = np.array(test_data.iloc[i], dtype=float)
        list_temp = []
        #list_temp.append(1)
        list_temp.append(new_x[0])
        list_temp.append(new_x[1])
        list_temp.append(new_x[2])
        list_temp.append(new_x[3])
        list_temp1 = np.array(list_temp)
        y_predict_test.append( test(list_temp1,l_W_mat,B,type))
    list_max_index=[]
    main_Y_test=np.array(y_test)
    counter=0
    class1_true=0
    class2_true=0
    class3_true=0
    list_from_test1=[]
    list_from_test= []
    convutuin_matrix=[[0,0,0,0],[0,0,0,0],[0,0,0,0]]
    for i in range(len( y_predict_test)):
        result = np.where(y_predict_test[i] == np.amax(y_predict_test[i]))
        result1 = np.where(main_Y_test[i] == np.amax(main_Y_test[i]))
        #bta3t el test nafso
        if result1[0]==[0]:
            list_from_test1.append([1,0,0])
            convutuin_matrix[0][0]+=1
        if result1[0]==[1]:
            list_from_test1.append([0,1,0])
            convutuin_matrix[1][0] += 1
        if result1[0]==[2]:
            list_from_test1.append([0,0,1])
            convutuin_matrix[2][0] += 1
        #bta3t el natega el tl3t
        if result[0]==[0]:
            list_from_test.append([1,0,0])
        if result[0]==[1]:
            list_from_test.append([0,1,0])
        if result[0]==[2]:
            list_from_test.append([0,0,1])

    for i in range(len( y_predict_test)):
        if list_from_test[i]==list_from_test1[i]:
            counter+=1
            if list_from_test[i]==[1,0,0]:
                class1_true+=1
                convutuin_matrix[0][1] += 1
            if list_from_test[i] == [0, 1, 0]:
                class2_true += 1
                convutuin_matrix[1][1] += 1
            if list_from_test[i] == [0, 0, 1]:
                class3_true += 1
                convutuin_matrix[2][1] += 1
        if list_from_test[i]!=list_from_test1[i]:
           if list_from_test1 [i]==[1,0,0]:
               if list_from_test[i]==[0,1,0]:
                   convutuin_matrix[0][2]+=1
               if list_from_test[i] == [0, 0, 1]:
                   convutuin_matrix[0][3] += 1
           if list_from_test1[i] == [0, 1, 0]:
               if list_from_test[i] == [1, 0, 0]:
                   convutuin_matrix[1][2] += 1
               if list_from_test[i] == [0, 0, 1]:
                   convutuin_matrix[1][3] += 1

           if list_from_test1[i] == [0, 0, 1]:
              if list_from_test[i] == [0, 1, 0]:
                convutuin_matrix[2][2] += 1
              if list_from_test[i] == [1, 0, 0]:
                convutuin_matrix[2][3] += 1
    print("Confusion Matrix is=")
    print(convutuin_matrix[0])
    print(convutuin_matrix[1])
    print(convutuin_matrix[2])
    print ("Accuracy=")
    print((counter/60)*100)



