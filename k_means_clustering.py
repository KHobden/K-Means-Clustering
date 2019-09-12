#Cluster Analysis
#Kieran Hobden
#12-Sep-'19

import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import make_blobs

def new_centres(coords, labels):
    #Find the centres of the new clusters by minimising the 1D distance
    new_centres = []
    #Permutes through possible lalels i.e. 0, 1, 2, 3
    for i in range(no_centres):
        #Array containing the indices of data with labels 0, 1, 2 or 3
        useful_indices = []
        for j in range(len(labels)):
            if labels[j] == i:
                #Appends indices that match the current i
                useful_indices.append(j)
        #Finds the centre through minimising the 1D distance
        average_x = np.sum(coords[useful_indices,0])/len(useful_indices)
        average_y = np.sum(coords[useful_indices,1])/len(useful_indices)
        centre = [average_x, average_y]
        new_centres.append(centre)
    #Python list must be converted to NumPy array for MatPlotLib
    new_centres = np.array(new_centres)
    return new_centres

def new_labels(coords, centres):
    #Find the nearest centroid to each point
    #Nearest centre labels each point with 0-3 denoting the centre it's nearest
    new_labels = []
    for i in coords:
        #4 valued array of the distances from a given point to each centre
        distances = []
        for j in centres:
            distances.append(np.linalg.norm(i - j))
        #Finds the shortest distance to a centre and appends the centre index
        new_labels.append(np.argmin(distances))
    return new_labels

def iterate(i, coords, centres, labels):
    for j in range(i):
        centres = new_centres(coords, labels)
        #Stops the iterations when no the clustering doesn't change
        labels_iterated = new_labels(coords, centres)
        if labels == labels_iterated:
            print("Optimum solution was obtained after %i iterations" %j)
            break
        else:
            labels = labels_iterated
        #Prints the centres and input data coloured to show the nearest centre
        plt.scatter(centres[:,0], centres[:,1], c="black")
        plt.scatter(coords[:,0], coords[:,1], c=labels, cmap='viridis', marker="x")
        plt.title("Clusters After %i Iterations" %(j+1))
        plt.show()
    

#Create random clusters of data
no_centres = 4
coords, labels = make_blobs(n_samples=200, centers=no_centres, cluster_std=2.0, random_state=10)

#Print the input data with which will use to find clusters
plt.scatter(coords[:,0], coords[:,1], marker="x")
plt.title("Input Data")
plt.show()

#Create random estimate of the coords of the centres
#Note scikit centres are within the range -10 < x,y < 10
np.random.seed(30)
centres = 20 * (np.random.random((no_centres,2)) - 0.5)
labels = new_labels(coords, centres)

#Find the centres of the new clusters by minimising the 1D distance
iterate(10, coords, centres, labels)
