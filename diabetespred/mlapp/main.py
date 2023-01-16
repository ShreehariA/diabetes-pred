import pandas as pd  # Importing Pandas to process Dataset as a dataframe
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score # for Accuracy score of an algorithms
from sklearn.metrics import confusion_matrix,recall_score # Evaluation Metrics
import joblib
from sklearn.ensemble import AdaBoostClassifier
from sklearn.ensemble import RandomForestClassifier
import csv
import math
import os
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

def train():
    f = []
    data = pd.read_csv("mlapp/diabetes.csv")  # Dataset converted as a pandas data frame
    acc = []
    x = data[['Pregnancies', 'Glucose', 'BloodPressure', # Input Parameters
              'SkinThickness', 'Insulin', 'BMI',
              'DiabetesPedigreeFunction', 'Age']]
    y = data['Outcome']

    # Splitting Train and Test data

    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.20, random_state=200)

    # Logistic Regression

    from sklearn.linear_model import LogisticRegression
    mod1 = LogisticRegression()
    mod1.fit(x, y)
    y_pred = mod1.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm=confusion_matrix(y_test,y_pred)
    rs=recall_score(y_test,y_pred)
    f.append('LogisticRegression\n')
    # f.append('accuracy:'+str(accuracy)+'\n')
    f.append('True positive:'+str(cm[0][0])+'\n')
    f.append('True negative:'+str(cm[0][1])+'\n')
    f.append('False positive:'+str(cm[1][0])+'\n')
    f.append('False negative:'+ str(cm[1][1])+'\n')
    f.append('Recall score:'+str(rs))
    f.append('\n\n')
    labels=['true Neg','False pos','false Neg','true pos']
    fig=plt.figure()
    labels=np.asarray(labels).reshape(2,2)
    print(labels)
    sns.heatmap(cm, annot=labels,fmt='')
    fig.savefig('mlapp/static/logic.jpg')
    print(cm)
    acc.append(accuracy)



    # Adaboost

    from sklearn.ensemble import AdaBoostClassifier
    ada = AdaBoostClassifier(n_estimators=50, learning_rate=1)
    ada.fit(x_train, y_train)
    y_pred = ada.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    rs = recall_score(y_test, y_pred)
    f.append('Adaboost\n')
    # f.write('accuracy:' + str(accuracy) + '\n')
    f.append('True positive:'+str(cm[0][0])+'\n')
    f.append('True negative:'+str(cm[0][1])+'\n')
    f.append('False positive:'+str(cm[1][0])+'\n')
    f.append('False negative:'+ str(cm[1][1])+'\n')
    f.append('Recall score:'+str(rs))
    f.append('\n\n')
    f.append('\n\n')
    fig = plt.figure()
    sns.heatmap(cm, annot=labels, fmt='')
    fig.savefig('mlapp/static/adaboost.jpg')
    print(cm)
    acc.append(accuracy)


    # RandomForest

    from sklearn.ensemble import RandomForestClassifier
    ran = RandomForestClassifier(n_estimators=50)
    ran.fit(x_train, y_train)
    y_pred = ran.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    rs = recall_score(y_test, y_pred)
    f.append('RandomForest\n')
    # f.write('accuracy:' + str(accuracy) + '\n')
    f.append('True positive:'+str(cm[0][0])+'\n')
    f.append('True negative:'+str(cm[0][1])+'\n')
    f.append('False positive:'+str(cm[1][0])+'\n')
    f.append('False negative:'+ str(cm[1][1])+'\n')
    f.append('Recall score:'+str(rs))
    f.append('\n\n')
    fig = plt.figure()
    sns.heatmap(cm, annot=labels, fmt='')
    fig.savefig('mlapp/static/randomforest.jpg')
    print(cm)
    acc.append(accuracy)

    # Decision Tree

    from sklearn.tree import DecisionTreeClassifier
    tree = DecisionTreeClassifier(criterion='gini')
    tree.fit(x_train, y_train)
    y_pred = tree.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    rs = recall_score(y_test, y_pred)
    f.append('Decision tree\n')
    # f.write('accuracy:' + str(accuracy) + '\n')
    f.append('True positive:'+str(cm[0][0])+'\n')
    f.append('True negative:'+str(cm[0][1])+'\n')
    f.append('False positive:'+str(cm[1][0])+'\n')
    f.append('False negative:'+ str(cm[1][1])+'\n')
    f.append('Recall score:'+str(rs))
    f.append('\n\n')
    fig = plt.figure()
    sns.heatmap(cm, annot=labels, fmt='')
    fig.savefig('mlapp/static/tree.jpg')
    print(cm)
    acc.append(accuracy)

    # GaussianNB

    from sklearn.naive_bayes import GaussianNB
    gauss = GaussianNB()
    gauss.fit(x_train, y_train)
    y_pred = gauss.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    rs = recall_score(y_test, y_pred)
    f.append('Gaussian NB\n')
    # f.write('accuracy:' + str(accuracy) + '\n')
    f.append('True positive:'+str(cm[0][0])+'\n')
    f.append('True negative:'+str(cm[0][1])+'\n')
    f.append('False positive:'+str(cm[1][0])+'\n')
    f.append('False negative:'+ str(cm[1][1])+'\n')
    f.append('Recall score:'+str(rs))
    f.append('\n\n')
    fig = plt.figure()
    sns.heatmap(cm, annot=labels, fmt='')
    fig.savefig('mlapp/static/nb.jpg')
    print(cm)
    acc.append(accuracy)

    # K-NN

    from sklearn.neighbors import KNeighborsClassifier
    knn = KNeighborsClassifier(n_neighbors=5, p=2)
    knn.fit(x_train, y_train)
    y_pred = knn.predict(x_test)
    accuracy = accuracy_score(y_test, y_pred)
    cm = confusion_matrix(y_test, y_pred)
    rs = recall_score(y_test, y_pred)
    f.append('KNN\n')
    # f.write('accuracy:' + str(accuracy) + '\n')
    f.append('True positive:'+str(cm[0][0])+'\n')
    f.append('True negative:'+str(cm[0][1])+'\n')
    f.append('False positive:'+str(cm[1][0])+'\n')
    f.append('False negative:'+ str(cm[1][1])+'\n')
    f.append('Recall score:'+str(rs))
    f.append('\n\n')
    fig = plt.figure()
    sns.heatmap(cm, annot=labels, fmt='')
    fig.savefig('mlapp/static/knn.jpg')
    print(cm)
    acc.append(accuracy)



    max_acc = acc.index(max(acc))
    algos = ["Logistic Regression","Adaboost","Random Forest"
    ,"Decision Tree","Gaussian Naive Bayes","k-Nearest Neighbour"]
    i=0
    print('----------------------------------------------')
    print("ACCURACY SCORES OF MACHINE LEARNING ALGORIHMS")
    print('----------------------------------------------')
    final = []
    for algo in algos:
        print(algo+":"+str(("{:.5f}".format(acc[i]))))
        final.append(algo+":"+str(("{:.5f}".format(acc[i]))))
        i=i+1
    print('----------------------------------------------')
    print(acc)
    if(max_acc==0):
        joblib.dump(mod1,'diabetesmodel.joblib')
    elif(max_acc==1):
        joblib.dump(ada, 'diabetesmodel.joblib')

    elif(max_acc==2):
        joblib.dump(ran, 'diabetesmodel.joblib')
    elif(max_acc==3):
        joblib.dump(tree, 'diabetesmodel.joblib')

    elif(max_acc==4):
        joblib.dump(gauss, 'diabetesmodel.joblib')

    elif(max_acc==5):
        joblib.dump(knn, 'diabetesmodel.joblib')
    else:
        print("Error")
    return final,f,acc,algos
