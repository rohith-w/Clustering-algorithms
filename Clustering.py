# -*- coding: utf-8 -*-
"""
Created on Fri Mar 25 12:33:55 2022

@author: Rohit Wardole
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def prepareData(normalisation):
    
    dataFiles = ["veggies", "countries", "fruits", "animals"]
    X = []
    y = []
    
    for i in range(len(dataFiles)):
        dataFile = pd.read_csv(dataFiles[i], sep = " ", header = None)
        dataFile = dataFile.drop(0, axis = 1).values
        featureCategory = np.full(len(dataFile), i)
        
        X.extend(dataFile)
        y.extend(featureCategory)
    
    X = np.reshape(X, (327,300))
    y = np.reshape(y, (327,))
    
    if normalisation:
        X = X / np.linalg.norm(X)
        
    return X,y

# To calculate K-Means 
def calculateEuclideanDistance(data,centroids):
    shortest_distance = []
    for i in range(len(centroids)):
        distance = np.sum((data - centroids[i])**2)
        shortest_distance.append(distance)
        
    centroid = np.argmin(shortest_distance)    
    return centroid

# To calculate K-Medians
def calculateManhattanDistance(data,centroids):
    shortest_distance = []
    for i in range(len(centroids)):
        distance = np.sum(abs(data - centroids[i]))
        shortest_distance.append(distance)
        
    centroid = np.argmin(shortest_distance)    
    return centroid

def clusteringAlgorithm(X, y, k, k_means):
    
    clusters = []
    centroids = []

    # Step 2: Select K random points from the data as centroids
    idx = np.random.choice(len(X), k, replace = False)
    for i in idx:
        centroids.append(X[i])        
       
    for n_iterations in range(100):
        
        # Step 3: Assign all the points to the closest cluster centroid
        temp_clusters = []
        for i in range(k):
            temp_clusters.append([])
        
        for i, data in enumerate(X):
            # If K_means is true then use Euclidean Distance else Manhattan Distance for K_medians
            if k_means:
                centroid = calculateEuclideanDistance(data,centroids)
            else:
                centroid = calculateManhattanDistance(data,centroids)
                
            temp_clusters[centroid].append(i)
            
        clusters = temp_clusters  

        # Step 4: Recompute the centroids of newly formed clusters
        old_centroids = centroids
        
        new_centroids = np.zeros((k,X.shape[1]))
        
        for i, cluster in enumerate(clusters):
            # If K_means is true then use calculate Mean else calculate Median for K_medians
            if k_means:
                new_centroid = np.mean(X[cluster], axis = 0)
            else:
                new_centroid = np.median(X[cluster], axis = 0)
                
            new_centroids[i] = new_centroid
        
        distances = []
        
        for i in range(k):
            distance = np.sum((old_centroids[i] - new_centroids[i])**2)
            distances.append(distance)
            
        if sum(distances) == 0:
            break

    return clusters

def trueLabels(clusts,y):
    cluster_labels = []
    for i in range(len(clusts)):
        
        cluster = clusts[i]
        labels = y
      
        for j in range(len(cluster)):
            
            cluster_labels.append(labels[cluster[j]])

    return cluster_labels

def bCubed(clusters,y):
    
    precisions = []
    recalls = []
    f_scores = []
    
    clusts = clusters
    
    allClustersLabels = trueLabels(clusts,y)
    
    for i in range(len(clusters)):
        cluster = clusters[i]
        labels = y
        
        cluster_labels = []
        for j in range(len(cluster)):
            cluster_labels.append(labels[cluster[j]])
            
        for i in range(len(cluster_labels)):
            value = cluster_labels[i]
            thisCluster = np.count_nonzero(cluster_labels==value)
            allCluster  = np.count_nonzero(allClustersLabels==value)
            
            precision = thisCluster / len(cluster_labels)
            precisions.append(precision)
            
            recall = thisCluster / allCluster
            recalls.append(recall)
            
            f_score = (2*recall*precision) / (recall + precision)
            f_scores.append(f_score)
    
    return precisions,recalls,f_scores

def plotGraph(cluster_number, precision, recall, f_score):
    
    plt.plot(cluster_number, precision,  marker='+', linestyle='-', color='r', label='Preicison') 
    plt.plot(cluster_number, recall,   marker='+', linestyle='-', color='g', label='Recall') 
    plt.plot(cluster_number, f_score,    marker='+', linestyle='-', color='b', label='F-Score') 
    plt.ylabel('Results')
    plt.xlabel('Cluster numbers') 
    plt.title('B-CUBED Results')
    plt.legend()
    plt.show()
        
def taskHelper(k_means, normalisation):
    
    cluster_numbers = []
    precision = []
    recall = []
    f_score = []
    
    X,y = prepareData(normalisation)
    
    # Step 1: Choose the number of clusters K
    for k in range(1,10):
        
        cluster_numbers.append(k)
        clusters = clusteringAlgorithm(X,y,k,k_means)

        prec,rec,fs = bCubed(clusters, y)
        precision.append(np.average(prec))
        recall.append(np.average(rec))
        f_score.append(np.average(fs))
    
    plotGraph(cluster_numbers, precision, recall, f_score)
    print("clusters numbers",   cluster_numbers)
    print("precisions", np.round(precision, 2))
    print("recalls", np.round(recall, 2))
    print("F-score", np.round(f_score, 2))      
    
if __name__ == '__main__':    
    np.random.seed(44)
    print("===========Answer 3 (B-Cubed for K means)============")
    taskHelper(k_means = True, normalisation = False)
    print("=======Answer 4 (B-Cubed for K means Normalised)=====")
    taskHelper(k_means = True, normalisation = True)
    print("===========Answer 5 (B-Cubed for K medians)==========")
    taskHelper(k_means = False, normalisation = False)
    print("=====Answer 6 (B-Cubed for K medians Normalised)=====")
    taskHelper(k_means = False, normalisation = True)