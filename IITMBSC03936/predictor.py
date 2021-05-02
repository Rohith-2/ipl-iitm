### Custom definitions and classes if any ###
import pickle
import numpy as np
from sklearn import preprocessing
import pandas as pd
import os

model_filename   = os.path.join(os.getcwd(), 'le_stadium.pkl')
model_filename_1 = os.path.join(os.getcwd(), "le_teams.npy")
model_filename_2 = os.path.join(os.getcwd(), "le_players.npy")

le_stadium = pickle.load(open(model_filename, 'rb'))

le_teams = preprocessing.LabelEncoder()
le_teams.classes_ = np.load(model_filename_1,allow_pickle=True)

le_players = preprocessing.LabelEncoder()
le_players.classes_ = np.load(model_filename_2,allow_pickle=True)

def inpu(X):
    pc = le_players.classes_.shape[0]-1
    X_m = np.zeros(len(X))
    for i in range(len(X)):
        if (i==0):
            X_m[i] = le_stadium[X[i]]
        elif(i==1):
            X_m[i] = X[1]
        elif(i==2 or i==3):
            if(X[i]=="Punjab Kings"):
                X_m[i] = np.where(le_teams.classes_ == "Kings XI Punjab")[0][0]
            else:
                X_m[i] = np.where(le_teams.classes_ == X[i])[0][0]
        elif(i>3):
            if X[i] == "Kyle Jamieson":
                X[i] = "KA Jamieson"
            if X[i] == "G Maxwell":
                X[i] = "GJ Maxwell"
            if X[i] == "RG Sharma":
                X[i] = "R Sharma"   
            if (X[i] not in le_players.classes_):
                X[i] = pc
                pc+=1
            else:
                X_m[i] = np.where(le_players.classes_ == X[i])[0][0]
            
    return X_m      

def predictRuns(testInput):

    max_bat  = 9
    max_bowl = 6


    data = pd.read_csv(testInput)
    X    = data.values.tolist()[0]
    X[4] = X[4].split(',')
    X[5] = X[5].split(',')

    for j in range(max_bat-len(X[4])):
        X[4].extend(["_"])
    for j in range(max_bowl-len(X[5])):
        X[5].extend(["_"])  

    X.extend(X[4])
    X.pop(4)
    X.extend(X[4])
    X.pop(4) 

    x_t = inpu(X)
    xx = np.where(x_t==le_players.classes_.shape[0]-1,-1, x_t).astype(int)
    test = xx.reshape(-1,1).T

    filename   = os.path.join(os.getcwd(), '*/.sav')
    loaded_model = pickle.load(open(filename, 'rb'))
    
    prediction = int(np.ceil(loaded_model.predict(test)[0])) - 4

    return prediction
