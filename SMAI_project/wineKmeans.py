import pandas as pd
from random import sample
from math import sqrt
from numpy import mean
import numpy as np
import copy
import csv

MAX_ITER = 100

def readCSV(data):
    dataList=[]
    classList=[]
    with open(data, "rb") as f_obj:
        reader = csv.reader(f_obj)
        for row in reader:
            if '?' in row:
                continue
            if row[0]=="1":
                classList.append(0)
            elif row[0]=="2":
                classList.append(1)
            else:
                classList.append(2)
            
            row = row[1:]
            row = map(float,row)    
            dataList.append(row)
            
    c = len(np.unique(classList))
    return dataList,classList,c



def initializeCenters(df, k):
    random_indices = sample(range(len(df)), k)
    centers = [list(df[idx]) for idx in random_indices]
    # centers = [list(df.iloc[idx]) for idx in random_indices]   made 1 change
    # print("Random Indices : " + str(centers))
    return centers


def computeCenter(df, k, cluster_labels):
    cluster_centers = list()
    data_points = list()
    for i in range(k):
        for idx, val in enumerate(cluster_labels):
            if val == i:
                data_points.append(list(df[idx]))
                #data_points.append(list(df.iloc[idx]))
        # print data_points
        cluster_centers.append(map(mean, zip(*data_points)))
    return cluster_centers


def euclidean_distance(x, y):
    summ = 0
    # print x
    x = map(float,x)
    y = map(float,y)
    for i in range(len(x)):
        term = (x[i] - y[i])**2
        summ += term
    return sqrt(summ)


def assignCluster(df, k, cluster_centers):
    cluster_assigned = list()
    for i in range(len(df)):
        distances = [euclidean_distance(list(df[i]), center) for center in cluster_centers]
        # print distances
        # distances = [euclidean_distance(list(df.iloc[i]), center) for center in cluster_centers] made 2nd change
        min_dist, idx = min((val, idx) for (idx, val) in enumerate(distances))
        cluster_assigned.append(idx)
    return cluster_assigned


def kmeans(df, k, class_labels):
    cluster_centers = initializeCenters(df, k)
    curr = 1
    
    while curr <= 10:
        cluster_labels = assignCluster(df, k, cluster_centers)
        # print (cluster_labels)
        prev_centers = copy.deepcopy(cluster_centers)
    #     print("Previous Cluster Centers: \n")
        cluster_centers = computeCenter(df, k, cluster_labels)
        # print cluster_centers
        curr += 1
#         print("Cluster 0: " + str(euclidean_distance(prev_centers[0], cluster_centers[0])))
#         print("Cluster 1: " + str(euclidean_distance(prev_centers[1], cluster_centers[1])))
    
    return cluster_labels, cluster_centers



def check(gt,index):
    # print gt,index
    n = len(gt)
    uY = list(np.unique(gt))
    nClass = len(uY)
    Y0 = np.zeros((n,1));

    if nClass != max(gt):
        for i in range(nClass):
            for j in range(len(gt)):
                if gt[j] == uY[i]:
                    Y0[j]=i

        gt = Y0

    uY = list(np.unique(index))
    nclass = len(uY)
    predY0 = np.zeros((n,1))
    # print uY,nclass,predY0

    if nClass != max(index):
        for i in range(nclass):
            for j in range(len(index)):
                if index[j] == uY[i]:
                    index[j]=i

        index = predY0

    Lidx = list(np.unique(gt))
    classnum = len(Lidx)
    predLidx = list(np.unique(index))
    pred_classnum = len(predLidx)

    # print index,pred_classnum

    correnum = 0
    for i in range(pred_classnum):
        s = []
        incluster = []
        for j in range(len(index)):
            if index[j] == predLidx[i]:
                s.append(j)

        for j in s:
            incluster.append(gt[j])
        maxIncluster = max(incluster)
        h = list(range(1,maxIncluster+1))
        inclunub = np.histogram(incluster,h)
        inclu = inclunub[0]
        if len(inclu) == 0:
            inclu = [0]
        correnum = correnum*1.0 + max(inclu);
    purity = (correnum)/len(index)
    return purity

#----------------------------------------------------------------------
if __name__ == "__main__":
   
    data,class_labels,k = readCSV("wine.csv")
    #########################################
    labels, centers = kmeans(data, k, class_labels)

    print("Number of data points in Cluster 0: " + str(labels.count(0)))
    print("Number of data points in Cluster 1: " + str(labels.count(1)))
    print("Number of data points in Cluster 1: " + str(labels.count(2)))

    purity = check(class_labels,labels)
    print "purity: ",float(purity)