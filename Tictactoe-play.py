#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon May 22 16:23:20 2017

@author: nissenadam
"""
VERBOSE = 0
print "asd"
import numpy as np
#np.random.seed(1)
length = 9 #dont't touch, length nine is necessary for tic tac toe
hidlaysiz = 16
trainsize = 1
board = np.array([[0,1,1],[1,0,0],[0,1,1]])
'''print board'''
#giving synapses between nodes(neurons) layers random initial strength values
#these are what changes as the network learns
#5 is the size of the two hidden layers
#7 is input data size, 1 is output data size
syn2 = 2*np.random.random((hidlaysiz,1)) - 1
syn1 = 2*np.random.random((hidlaysiz,hidlaysiz)) - 1
syn0 = 2*np.random.random((length,hidlaysiz)) - 1
#the important probability curve that you took forever to understand
#you still don't fully understand why it works
#what the fuck
def sigmoid(x,deriv=False):
    if (deriv==True):
        return x*(1-x)
    return 1/(1+np.exp(-x))

#function that iterates one generation and trains synapses
#X is input data and y is correct output
def train_generation(X,y,syn0,syn1,syn2,it,it_):
    #forward propagation
    l0 = X
    l1 = sigmoid(np.dot(l0,syn0))
    l2 = sigmoid(np.dot(l1,syn1))
    l3 = sigmoid(np.dot(l2,syn2))
    
    #how far was NN guess from correct answer, now with two layers
    l3_error = y - l3
    if (it%100 == 0):
        if (it_%1000 == 0):
            print np.mean(np.abs(l3_error))
            
    #delta is error multiplied by deriv of sigmoid
    #deriv of sigmoid is "how sure" weights are about answer, so if it is more
    #sure, the delta is less and the weight changes less
    l3_delta = l3_error * sigmoid(l3,True)
    
    #this calculation of l1 and l2 error is back propagation
    l2_error = l3_delta.dot(syn2.T)
    l2_delta = l2_error * sigmoid(l2,True)
    l1_error = l2_delta.dot(syn1.T)
    l1_delta = l1_error * sigmoid(l1,True)
    
    #adjusting weights by delta
    syn2 += l2.T.dot(l3_delta) 
    syn1 += l1.T.dot(l2_delta)
    syn0 += l0.T.dot(l1_delta)
    
    return syn0,syn1,syn2 

def propogate(l0,syn0,syn1,syn2):
    #l0 = [[0,1,1,1,0,0,0,1,1]]
    l1 = sigmoid(np.dot(l0,syn0))
    l2 = sigmoid(np.dot(l1,syn1))
    l3 = sigmoid(np.dot(l2,syn2))
    return l3

def isgameover (board):
    num = 0
    for p in range(3):
        if (board[p] == board[p+3] == board[p+6]):
            num = board[p]
        if (board[p*3] == board[p*3+1] == board[p*3+2]) and (board[p*3] != 0):
            num = board[p*3]
    if (board[0] == board[4] == board[8]) and (board[0] != 0):
        num = board[0]
    if (board[2] == board[4] == board[6])and (board[2] != 0):
        num = board[2]
    return num
    
def create_data():
    print '1'
    dnumber = 0
    end = 0
    test_set = np.zeros((trainsize,9),dtype = int)
    result_set = np.zeros((trainsize,1),dtype = int)
    for a in range (trainsize):
        #board
        game = np.zeros(9,dtype = int)
        #will be set to random point in the game to be fed to neural net
        game_point = np.zeros(9,dtype = int)
        #board for random outcomes that are played through and to see who won in each random play through
        random_outcome = np.zeros(9,dtype = int)
        end = 0
        moves = np.random.permutation(9)
        move = 0
        while (end == 0 and move <= 8):
            if (move%2 == 0):
                game[moves[move]] = 1
            if (move%2 == 1):
                game[moves[move]] = 2
            move += 1
            end = isgameover(game)
        show_point = np.random.randint(1,move-1)
        for x in range(show_point):
            if (x%2 == 0):
                game_point[moves[x]] = 1
            if (x%2 == 1):
                game_point[moves[x]] = 2
        print game_point.reshape(3,3)
        print game_point
        for x in range (5):
            new_moves = np.random.permutation(moves[show_point:])
            random_outcome = 1* game_point
            if (VERBOSE > 1): 
                print new_moves, random_outcome
            move = 0
            end = 0
            print
            while (end == 0 and move <= (8-show_point)):
                if (move%2 == 0):
                    random_outcome[new_moves[move]] = 1
                if (move%2 == 1):
                    random_outcome[new_moves[move]] = 2
                move += 1
                end = isgameover(random_outcome)
                print end
            print random_outcome.reshape(3,3) 
            
        if (end == 0):
            result_set [dnumber] = 0.5
        if (end == 1):
            result_set [dnumber] = 1
        if (end == 2):
            result_set [dnumber] = 0
                       
        dnumber +=1

    #return test_set,result_set
     
"""for x in range(400):
    inputs,results = create_data()
    for y in range(4000):
        syn0,syn1,syn2 = train_generation(inputs,results,syn0,syn1,syn2,x,y)"""
create_data()

'''for x in range (601):     
    num = 0
    y = []
    input_data = np.zeros((trainsize,length))
    #creates input datas of finished boards (5 ones and 4 zeros for a finished board)
    for i in range(trainsize):
        ones = np.random.permutation(np.arange(9))[:5]
        for pooop in range(5):
            input_data[i][ones[pooop]] = 1
    #identifies if the board has a three in a row and if not
    #stays classified as a draw (num = 0 vs num = 1)
    for q in range (trainsize):
        num = 0
        for p in range(3):
            if (input_data[q][p] == input_data[q][p+3] == input_data[q][p+6]):
                num = 1
            if (input_data[q][p*3] == input_data[q][p*3+1] == input_data[q][p*3+2]):
                num = 1
        if (input_data[q][0] == input_data[q][4] == input_data[q][8]):
            num = 1
        if (input_data[q][2] == input_data[q][4] == input_data[q][6]):
            num = 1
        #inputs result into training data correct answers
        y.append(num)
    y = np.array(y)
    y = y.reshape((trainsize,1))
    for z in range(1000):
        syn0,syn1,syn2 = train_generation(input_data,y,syn0,syn1,syn2,x,z)
            
'''
'''print round(l3)
print l3
print syn0
print ''
print syn1
print '' 
print syn2
'''