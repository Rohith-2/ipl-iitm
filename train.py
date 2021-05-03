from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestRegressor
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn import preprocessing
import glob
from sklearn.metrics import r2_score
import pickle

data = pd.read_csv('all_matches.csv')
data.drop(columns=["match_id", "season","start_date"], inplace=True)
data_n1 = data.to_numpy()
players = np.append(np.unique(np.append(np.unique(np.append(np.unique(data['striker']),np.unique(data['non_striker']))),np.unique(data['bowler']))),"Shahrukh Khan")

def create_train_data(data_n):
    T = [[]]
    for i in data_n:
        t = []
        t.extend([i[0], i[1], i[3], i[4], [], []])
        if(float(i[2]) < 6.0):
            if(i[5] not in t[4]):
                t[4].extend([i[5]])
            if(i[6] not in t[4]):
                t[4].extend([i[6]])
            if(i[7] not in t[5]):
                t[5].extend([i[7]])

            if(t[0] not in T[-1]):
                T.append(t)
    #         if(t not in T):
    #             T.append(t)


            if(t[0] in T[-1]):
                if(t[1] not in T[-1]):
                    T.append(t)

                if(t[1] in T[-1]):
                    for k in range(len(t[4])):
                        if(t[4][k] not in T[-1][4]):
                            T[-1][4].extend([t[4][k]])
                    for k in range(len(t[5])):
                        if(t[5][k] not in T[-1][5]):
                            T[-1][5].extend([t[5][k]])
    T.pop(0)
    return T

input_data = create_train_data(data_n1)

def score_list(ddd1):
    sum_ = ddd1[0][8]+ddd1[0][9]
    yy = []
    for i in range(1, len(ddd1)):
        if(int(ddd1[i][2])<6.0):
            if(ddd1[i][1] == ddd1[i-1][1] and ddd1[i][0] == ddd1[i-1][0]):
                sum_ += ddd1[i][8]+ddd1[i][9]

            elif(ddd1[i][1] != ddd1[i-1][1] and ddd1[i][0] != ddd1[i-1][0]):
                yy.append(sum_)
                sum_ = ddd1[i][8]+ddd1[i][9]

            elif(ddd1[i][1] != ddd1[i-1][1] and ddd1[i][0] == ddd1[i-1][0]):
                yy.append(sum_)
                sum_ = ddd1[i][8]+ddd1[i][9]

            elif(ddd1[i][1] == ddd1[i-1][1] and ddd1[i][0] != ddd1[i-1][0]):
                yy.append(sum_)
                sum_ = ddd1[i][8]+ddd1[i][9]
            
            else:
                print(ddd1[i])
         
    yy.append(sum_)
    return yy

y = np.array(score_list(data_n1))
ylist = y.tolist()
ylist.extend([26,32,46,45,55,39,32,55,54,45,32,59,21,45,25,50,51,65,51,56,37,42,36,43,39,50,47,49,45,67,49,36,49,58,42,57,39,63])
y = np.array(ylist)

max_bat=0
max_bowl=0
for i in input_data:
    if len(i[4])>max_bat:
        max_bat = len(i[4])
    if len(i[5])>max_bowl:
        max_bowl = len(i[5])

for i in input_data:
    for j in range(max_bat-len(i[4])):
        i[4].extend(["_"])
    for j in range(max_bowl-len(i[5])):
        i[5].extend(["_"])

T = input_data

path = "Archive/" # use your path
all_files = glob.glob(path + "/*.csv")
li = []
for filename in all_files:
    df = pd.read_csv(filename, index_col=None, header=0)
    li.append(df)
frame = pd.concat(li, axis=0, ignore_index=True)

input_1 = frame.values.tolist()
for i in input_1:
    i[4] = i[4].split(",")
    i[5] = i[5].split(",")
for i in input_1:
    for j in range(max_bat-len(i[4])):
        i[4].extend(["_"])
    for j in range(max_bowl-len(i[5])):
        i[5].extend(["_"])

T.extend(input_1[::-1])

for i in range(len(T)):
        T[i][4].extend(T[i][5])
        T[i].pop(5)
    
    #Mearging bowlers and bats with total
for i in range(len(T)):
    T[i].extend(T[i][4])
    T[i].pop(4)

X = pd.DataFrame(T)

#le_stadium = preprocessing.LabelEncoder()
#le_stadium.fit(X[0])
#X[0] = le_stadium.transform(X[0])

le_teams = preprocessing.LabelEncoder()
le_teams.fit(X[2])
X[2] = le_teams.transform(X[2])
X[3] = le_teams.transform(X[3])

X_modified = []
for i in range(4,X.shape[1]):
    X_modified.extend(X[i])

p = players.tolist()
p.extend(["Jhye Richardson","Chetan Sakariya","Fabian Allen",'Riley Meredith',"D Malan",'J Saxena','L Meriwala',"_"])
players = np.array(p)
le_players = preprocessing.LabelEncoder().fit(players)

X.replace("F Du Plessis","F du Plessis",inplace=True)
X.replace("J Buttler","JC Buttler",inplace=True)
X.replace("P Cummins","PJ Cummins",inplace=True)
X.replace("Kyle Jamieson","KA Jamieson",inplace=True)
X.replace("RG Sharma","R Sharma",inplace=True)

for i in range(4,X.shape[1]):
    X[i] = le_players.transform(X[i])

le_stad = {'Arun Jaitley Stadium':0, 'Barabati Stadium':1, 'Brabourne Stadium':2,
       'Buffalo Park':3, 'De Beers Diamond Oval':4,
       'Dr DY Patil Sports Academy':5,
       'Dr. Y.S. Rajasekhara Reddy ACA-VDCA Cricket Stadium':6,
       'Dubai International Cricket Stadium':7, 'Eden Gardens':8,
       'Feroz Shah Kotla':9, 'Green Park':10,
       'Himachal Pradesh Cricket Association Stadium':11,
       'Holkar Cricket Stadium':12, 'JSCA International Stadium Complex':13,
       'Kingsmead':14, 'M Chinnaswamy Stadium':15, 'M.Chinnaswamy Stadium':15,
       'MA Chidambaram Stadium':16, 'MA Chidambaram Stadium, Chepauk':16,
       'MA Chidambaram Stadium, Chepauk, Chennai':16,
       'Maharashtra Cricket Association Stadium':17, 'Nehru Stadium':18,
       'New Wanderers Stadium':19, 'Newlands':20, 'OUTsurance Oval':21,
       'Punjab Cricket Association IS Bindra Stadium':22,
       'Punjab Cricket Association IS Bindra Stadium, Mohali':22,
       'Punjab Cricket Association Stadium, Mohali':22,
       'Rajiv Gandhi International Stadium':23,
       'Rajiv Gandhi International Stadium, Uppal':23,
       'Sardar Patel Stadium, Motera':24,
       'Saurashtra Cricket Association Stadium':25, 'Sawai Mansingh Stadium':26,
       'Shaheed Veer Narayan Singh International Stadium':27,
       'Sharjah Cricket Stadium':28, 'Sheikh Zayed Stadium':29,
       "St George's Park":30, 'Subrata Roy Sahara Stadium':31,
       'SuperSport Park':32, 'Vidarbha Cricket Association Stadium, Jamtha':33,
       'Wankhede Stadium':34, 'Wankhede Stadium, Mumbai':34,'Narendra Modi Stadium':35,'Arun Jaitley Stadium':36}

X_s = []
for i in X[0]:
    X_s.append(le_stad[i])

X[0] = X_s
X.replace(players.shape[0]-1,-1,inplace=True)

scalerfile = 'IITMBSC03936/le_stadium.pkl'
pickle.dump(le_stad, open(scalerfile, 'wb'))

np.save('IITMBSC03936/le_teams.npy', le_teams.classes_)
np.save('IITMBSC03936/le_players.npy', le_players.classes_)

Xtrain, Xtest, ytrain, ytest = train_test_split(X,y, test_size=0.1)

estimator = RandomForestRegressor(random_state=69)
param_grid = { 
            "n_estimators"      : range(140,200,5),
            "max_features"      : ["auto", "sqrt", "log2"],
            "min_samples_split" : range(80,180,5),
            "max_depth" : range(1,11,1)
            }

print(" ")
print("Starting Training..")
regr = RandomForestRegressor(max_depth=150,n_estimators=150, min_samples_leaf=1,random_state=69,min_samples_split=2,min_weight_fraction_leaf=0.000001)
regr.fit(X,y)

print("Training Complete..")

y1 = regr.predict(Xtest)
y_1 = np.array(y1)
y_1 = [int(i) for i in y_1]
ytest_1 = [int(i) for i in ytest]
x = r2_score(ytest_1,y_1)

print("Exporting Model..")
filename = 'IITMBSC03936/RF_model_1.sav'
pickle.dump(regr, open(filename, 'wb'))

print("Printing Eval : ")
print(((np.array(y_1) - np.array(ytest_1))**2))
print(x)
