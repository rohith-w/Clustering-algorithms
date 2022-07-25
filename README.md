<k-means & k-medians clustering>

The file clustering.py is used to generate output for this Project.

When the file is run __main__ function will execute all the answers for the given project.

Outputs will be consistent each time the program because the seed has been set to its default value.
Function List:
1) prepareData()
=> Data pre-proccessing is performed in this function so that the algorithm is ran on all the data objects.

2) calculateEuclideanDistance()
=> Used to calculate distance between two points when K-means algorithm is in process.

3) calculateManhattanDistance()
=> Used to calculate distance between two points when K-medians algorithm is in process.

4) clusteringAlgorithm()
=> A dynamic function which runs both K-means and K-medians algorithm based on the parameters provided to the function.
=> If K_means is true then use K-means logic else K_medians logic.

5) bCubed()
=> To calculate precision, recall and f-score for the clusters.

6) plotGraph()
=> Plots the graphs for all tasks 3-6

7) taskHelper()
=> It is the starting point of all the tasks.

They do need any parameters to be passed as all the arguments are preconfigured.

The data files were converted to .csv files.

The data files needs to be in the same directory as of python files.

Below is the output for the tasks 3-6
===========Answer 3 (B-Cubed for K means)============
clusters numbers [1,   2,   3,   4,   5,   6,   7,   8,   9]
precisions 		 [0.33 0.55 0.61 0.78 0.57 0.75 0.61 0.84 0.89]
recalls 		 [1.   0.88 0.72 0.9  0.45 0.49 0.37 0.51 0.48]
F-score 		 [0.47 0.66 0.58 0.81 0.45 0.55 0.42 0.59 0.57]
=======Answer 4 (B-Cubed for K means Normalised)=====
clusters numbers [1,   2,   3,   4,   5,   6,   7,   8,   9]
precisions 		 [0.33 0.54 0.6  0.54 0.56 0.63 0.85 0.6  0.72]
recalls 		 [1.   0.9  0.82 0.47 0.5  0.58 0.51 0.35 0.41]
F-score 		 [0.47 0.66 0.67 0.47 0.44 0.5  0.6  0.38 0.46]
===========Answer 5 (B-Cubed for K medians)==========
clusters numbers [1,   2,   3,   4,   5,   6,   7,   8,   9]
precisions 		 [0.33 0.4  0.49 0.7  0.69 0.73 0.85 0.81 0.87]
recalls 		 [1.   0.8  0.53 0.62 0.57 0.51 0.52 0.46 0.54]
F-score 		 [0.47 0.49 0.49 0.59 0.54 0.55 0.61 0.57 0.65]
=====Answer 6 (B-Cubed for K medians Normalised)=====
clusters numbers [1,   2,   3,   4,   5,   6,   7,   8,   9]
precisions 		 [0.33 0.59 0.64 0.67 0.76 0.79 0.57 0.8  0.81]
recalls 		 [1.   0.9  0.87 0.74 0.64 0.52 0.33 0.44 0.4 ]
F-score 		 [0.47 0.69 0.66 0.67 0.61 0.59 0.35 0.53 0.49]