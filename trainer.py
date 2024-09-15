import numpy as np

from perceptron import Perceptron

from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
 
import sys


def plot_cost_function(epochs, loss_functions): 
    """
    Generate 

    Returns
    -------
    None.

    """

    epochs = np.array(epochs)
    loss_functions = np.array(loss_functions)
    
    return None


def train_and_evaluate(X_train, 
                       y_train, 
                       X_test,
                       y_test, 
                       n_features, 
                       use_gd=False):
    """
    Train a Perceptron classifier and evaluate its performance.

    This function creates a Perceptron, trains it on the given training data,
    and evaluates its performance on the test data.
    """
    perceptron = Perceptron(n_features)

    # if use_gd:
    #     perceptron.fit_GD(X_train, y_tain)
    # else:
    #     perceptron.fit(X_train, y_train)

    perceptron.fit(X_train, y_train)
    
    print("Test data accuracy after fit method")
    
    y_test_pred = perceptron.predict(X_test)
    misclassified = np.sum(y_test_pred != y_test)
    #accuracy = (len(y_test) - misclassified) / len(y_test) * 100
    accuracy = accuracy_score(y_test, y_test_pred) * 100 
    print("Number of test data instances misclassified :", misclassified)
    print("Test data accuracy score: ", accuracy) 
    
    

    #y_pred = perceptron.predict(X_test)
    #misclassified = np.sum(y_pred != y_test)
    #accuracy = (len(y_test) - misclassified) / len(y_test) * 100
    #accuracy = accuracy_score(y_test, y_pred) * 100 
    
    
    print("--- accuracy after training/fit on test data: ", accuracy)
    
    #sys.exit(0)
    if use_gd : 
       print("Calling fitGD ...")
       perceptron.fitGD( X_train, y_train, max_epochs=1000)
       
       
       #print('--- train accuracy after fitGD ---')
       
       
       
       print("-- Test data accuracy after fitGD method")
       y_test_pred = perceptron.predict(X_test)
       misclassified = np.sum(y_test_pred != y_test)
       #accuracy = (len(y_test) - misclassified) / len(y_test) * 100
       accuracy = accuracy_score(y_test, y_test_pred) * 100 
       print("Number of test data instances misclassified :", misclassified)
       print("Test data accuracy score: ", accuracy) 
       
    

    #print(f"Misclassified samples: {misclassified}")
    #print(f"Accuracy: {accuracy:.2f}%")

    
    #if n_features == 2:
    #    pass

    #print('-------')
    #print("\t\t Original Algorithm")
    #print('-------\n\n')
    #print(f"{'Gradient Descent' if use_gd else 'Original Algorithm'}")
   
       #if use_gd : 
       #    perceptron.fitGD( X_train, y_train, max_epochs=100)
       #    y_pred = perceptron.predict(X_test)
       #    misclassified = np.sum(y_pred != y_test)
       #    accuracy = (len(y_test) - misclassified) / len(y_test) * 100
        
    
    #print(f"Misclassified samples: {misclassified}")
    #print(f"Accuracy: {accuracy:.2f}%")

    return perceptron, misclassified, accuracy

# TODO: Call this class to automate training
