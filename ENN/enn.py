import numpy as np
from sklearn.neighbors import KDTree
from scipy.spatial.distance import euclidean
from sklearn.base import ClassifierMixin
from sklearn.neighbors.base import NeighborsBase

class enn(ClassifierMixin, NeighborsBase):
    
    def __init__(self, k=3, distance_function = euclidean):
        self.k = k
        self.distance_function = distance_function    
        
    def buildDistanceMap (self, X, Y):
        classes = np.unique(Y)
        nClasses = len(classes)
        tree = KDTree(X)
        nRows = X.shape[0]

        TSOri = np.array([]).reshape(0,self.k)

        distanceMap = np.array([]).reshape(0,self.k)
        labels = np.array([]).reshape(0,self.k)

        for row in range(nRows):
            distances, indicesOfNeighbors = tree.query(X[row].reshape(1,-1), k = self.k+1)

            distances = distances[0][1:]
            indicesOfNeighbors = indicesOfNeighbors[0][1:]

            distanceMap = np.append(distanceMap, np.array(distances).reshape(1,self.k), axis=0)
            labels = np.append(labels, np.array(Y[indicesOfNeighbors]).reshape(1,self.k),axis=0)

        for c in classes:
            nTraining = np.sum(Y == c)
            labelTmp = labels[Y.ravel() == c,:]

            tmpKNNClass = labelTmp.ravel()
            TSOri = np.append(TSOri, len(tmpKNNClass[tmpKNNClass == c]) / (nTraining*float(self.k)))

        return distanceMap, labels, TSOri    
    
    
    def fit(self, X, Y):
        self.Y_train = Y
        self.X_train = X
        
        self.knnDistances, self.knnLabels, self.TSOri = self.buildDistanceMap(X, Y)

        self.classes = np.unique(Y)
        self.nClasses = len(self.classes)

        self.nTrainingEachClass = []
        for i,c in enumerate(self.classes):
            self.nTrainingEachClass.append(len(Y[Y == c]))

        
    def predict(self, X):
        y_pred = []
        
        for testingData in X:
            disNorm2 = []
            for row in self.X_train:
                dist = self.distance_function(row, testingData)
                disNorm2.append(dist)

            disNorm2 = np.array(disNorm2)
            sortIX = np.argsort(disNorm2)

            classNNTest = self.Y_train[sortIX][:self.k]

            hitNumKNN = []
            for c in self.classes:
                hitNumKNN.append(np.sum(classNNTest == c))

            TSENN = [0] * self.nClasses
            nTrainingNN = [0] * self.nClasses
            nSameTrainingNN = [0] * self.nClasses

            for i,c in enumerate(self.classes):
                mask = self.Y_train.ravel() == c
                testingMuDist = disNorm2[mask]
                trainingMuDist = self.knnDistances[mask][:,self.k-1]
                trainingMuClass = self.knnLabels[mask][:,self.k-1]
                difDist = testingMuDist - trainingMuDist

                C = difDist <= 0
                nTrainingNN[i] = np.sum(C)

                if nTrainingNN[i] > 0:
                    nSameTrainingNN[i] = np.sum(trainingMuClass[C] == c)

            for j in range(self.nClasses):
                deltaNumSame = nTrainingNN[j] - nSameTrainingNN[j]
                difTmp = np.array(nSameTrainingNN) / (np.array(self.nTrainingEachClass)*float(self.k))

                deltaNumDif = np.sum(difTmp) - nSameTrainingNN[j]/(self.nTrainingEachClass[j]*float(self.k))                    

                TSENN[j] = (deltaNumSame + hitNumKNN[j] - self.TSOri[j] * self.k) / ((self.nTrainingEachClass[j]+1)*self.k) - deltaNumDif    

            y_pred.append(self.classes[np.argmax(TSENN)])

        return y_pred    
        
    
