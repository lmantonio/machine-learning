import matplotlib.pyplot as plt
import matplotlib.collections
import numpy as np
import matplotlib.tri as tri
import matplotlib.style
import matplotlib as mpl
import os #used to save the plot in the designated directory

#------------------------------------------------------------------------------
#linear regression for prediction
#
# Code pipeline:
# (1) import trainning set or append new data
# (2) adjust hypothesis function for prediction
# (3) output prediction on demand
#------------------------------------------------------------------------------

def get_trainingSet(fileName):
    
    trainingSet_rowSize = 0
    trainingSet_colSize = 0

    f = open(fileName)
    while True:
        line = f.readline()  
        if not line:
            break
        else:
            line_str = line.split()     
            trainingSet_rowSize = trainingSet_rowSize+1
            trainingSet_colSize = len(line_str)
    f.close()

    #X = np.zeros((trainingSet_rowSize,trainingSet_colSize-1))
    matrix_features = np.zeros((trainingSet_rowSize,trainingSet_colSize-1))
    #Y = np.zeros((trainingSet_rowSize,1))
    vector_outputs = np.zeros((trainingSet_rowSize,1))

    f = open(fileName)
    row = 0
    while True:
        line = f.readline()  
        if not line:
            break
        else:
            line_str = line.split()     
            for i in range(0,trainingSet_colSize-1):
                matrix_features[row][i] = float(line_str[i])
        
            vector_outputs[row] = float(line_str[trainingSet_colSize-1])
            row = row+1

    f.close()

    return matrix_features,vector_outputs

def log_modelCoef(point_theta_updated,input_scaling_mean,input_scaling_stdv,fileName):
    trainingSet_colSize = len(point_theta_updated)
    f = open(fileName,"w")
    for i in range(0,trainingSet_colSize):    
        f.write(str(point_theta_updated[i][0]))
        f.write(' ')
    f.write('\n')
    for i in range(0,trainingSet_colSize):    
        f.write(str(input_scaling_mean[i][0]))
        f.write(' ')
    f.write('\n')
    for i in range(0,trainingSet_colSize):    
        f.write(str(input_scaling_stdv[i][0]))
        f.write(' ')
    f.close()

    return 0

def set_featureScaling(matrix_features):

    (numTrainingExamples, numFeatures) = matrix_features.shape
    matrix_features_scaled = np.zeros((numTrainingExamples,numFeatures))
    matrix_features_scaled[:,0] = 1.0

    features_mean = np.zeros((numFeatures,1))
    features_stdv = np.ones((numFeatures,1))    

    for i in range(1,numFeatures):
        
        features_mean[i][0] = np.mean(matrix_features[:,i])
        features_stdv[i][0] = np.std(matrix_features[:,i])
        
        for j in range(0,numTrainingExamples):
            matrix_features_scaled[j][i] = (matrix_features[j][i]-features_mean[i][0])/features_stdv[i][0]                

    return matrix_features_scaled,features_mean,features_stdv

def set_costFunction(point_theta,matrix_features,vector_outputs):
    
    (numTrainingExamples, numFeatures) = matrix_features.shape
    grad_costFunction = np.zeros((numFeatures,1))
    #--------------------------------------------------------------------------
    #cost function calculation    
    #--------------------------------------------------------------------------
    sum = 0.0
    for i in range(0,numTrainingExamples):
        h = np.dot(matrix_features[i],point_theta)
        sum = sum + (h-vector_outputs[i])**2
    
    costFunction = 0.5*sum/numTrainingExamples
    
    #--------------------------------------------------------------------------
    #cost function gradient calculation    
    #--------------------------------------------------------------------------
    sum = 0.0
    for j in range(0,numFeatures):

        sum = 0.0
        for i in range(0,numTrainingExamples):
            h = np.dot(matrix_features[i],point_theta)
            sum = sum + (h-vector_outputs[i])*matrix_features[i][j]
        
        grad_costFunction[j][0] = sum/numTrainingExamples

    return costFunction, grad_costFunction


def set_learningRate(point_theta0,matrix_features,vector_outputs,descentDirection):
    
    maxIter = 1000
    alp0 = 0
    alp1 = 1
    (numTrainingExamples, numFeatures) = matrix_features.shape

    (costFunction,grad_costFunction) = set_costFunction(point_theta0,matrix_features,vector_outputs)        
    directionDerivate0 = np.matmul(grad_costFunction.transpose(),descentDirection)
        
    tol = 1.0e-2*np.linalg.norm(directionDerivate0)
       
    for i in range(0,maxIter):
        point_theta1 = point_theta0 + alp1*descentDirection
        (costFunction,grad_costFunction) = set_costFunction(point_theta1,matrix_features,vector_outputs)
        directionDerivate1 = np.matmul(grad_costFunction.transpose(),descentDirection)        
        if directionDerivate0*directionDerivate1 <= 0:
            break
        else:
            alp1 = 2*alp1
    
    for i in range(0,maxIter):
        alp2 = alp0-directionDerivate0*(alp1-alp0)/(directionDerivate1-directionDerivate0)
        point_theta2 = point_theta0 + alp2*descentDirection
        (costFunction,grad_costFunction) = set_costFunction(point_theta2,matrix_features,vector_outputs)
        directionDerivate2 = np.matmul(grad_costFunction.transpose(),descentDirection)      
        
        if np.linalg.norm(directionDerivate2) <= tol:
            alp = alp2
            break
        else:
            if directionDerivate0*directionDerivate2<0:
                alp1 = alp2
                directionDerivate1 = directionDerivate2
            else:
                alp0 = alp2
                directionDerivate0 = directionDerivate2

    alp = alp2
    return alp

def gradDescent(matrix_features,vector_outputs):

    (numTrainingExamples, numFeatures) = matrix_features.shape
    tol = 1e-6
    maxIter = 1000000

    point_theta_min = np.zeros((numFeatures,1))
    point_theta0 = np.zeros((numFeatures,1))
    point_theta1 = np.zeros((numFeatures,1))

    (costFunction,grad_costFunction0) = set_costFunction(point_theta0,matrix_features,vector_outputs)

    if np.linalg.norm(grad_costFunction0) < tol:
        point_theta_min = point_theta0
    else:
        descentDirection0 = -grad_costFunction0
        learningRate = set_learningRate(point_theta0,matrix_features,vector_outputs,descentDirection0)
        point_theta1 = point_theta0+learningRate*descentDirection0
        
        for i in range (0,maxIter):
            (costFunction,grad_costFunction1) = set_costFunction(point_theta1,matrix_features,vector_outputs)
            if np.linalg.norm(grad_costFunction1) <= tol:
                point_theta_min = point_theta1
                break
            else:
                descentDirection1 = -grad_costFunction1
                learningRate = set_learningRate(point_theta1,matrix_features,vector_outputs,descentDirection1)
                point_theta2 = point_theta1+learningRate*descentDirection1
                point_theta1 = point_theta2

    return point_theta_min

#------------------------------------------------------------------------------
#import trainning set
#------------------------------------------------------------------------------
X,Y = get_trainingSet('trainingset.txt')

X,meanX,stdX = set_featureScaling(X)


#------------------------------------------------------------------------------
#learning algorithm - update hypothesis for prediction
#------------------------------------------------------------------------------
modelcoef = gradDescent(X,Y)
log_modelCoef(modelcoef,meanX,stdX,"modelcoef.txt")
print('Model training finished - model coeficients updated')



#e = np.matmul(np.linalg.inv(np.matmul(X.transpose(),X)),np.matmul(X.transpose(),Y))

#print(abs(b-e))

#print(e)

