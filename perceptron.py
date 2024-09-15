import numpy as np

class Perceptron:
    """
    A Perceptron classifier.

    This class implements both the original Perceptron learning algorithm
    [Rosenblatt 1958] and a variant using the gradient descent optimizaiton.

    Attributes
    ----------
    weights : numpy.ndarray
        The weight vector of the Perceptron.
    bias : float
        The bias term of the Perceptron.
    """

    def __init__(self, num_inputs, learning_rate=0.001):
        """
        Initialize the Perceptron with random weights and a bias, and set
        the learning rate.

        Parameters
        ----------
        num_inputs : int
            The number of input features.
        learning_rate : float
            The learning rate (default is 0.01).
        """
        self.weights = np.random.uniform(-1, 1, num_inputs)
        self.bias = np.random.uniform(-1, 1)
        self.learning_rate = learning_rate
        
                
        self.accuracies_ = []
        self.loss_function_ = [] 
        self.epochs_ = []
    
    def get_loss_function_data(self): 
        """
        
        Returns
        -------
        TYPE
            DESCRIPTION.

        """
        return np.array(self.epochs_, self.loss_function_) 
  
    
  
    
    
    def forward(self, inputs):
        """
        Compute the forward pass of the Perceptron.

        Parameters
        ----------
        inputs : numpy.ndarray
            The input samples.

        Returns
        -------
        numpy.ndarray
            The output of the Perceptron before thresholding.
        """
        return np.dot(inputs, self.weights) + self.bias

    def predict(self, inputs):
        """
        Predict the class labels for the input samples.

        Parameters
        ----------
        inputs : numpy.ndarray
            The input samples.

        Returns
        -------
        numpy.ndarray
            The predicted class labels (-1 or 1).
        """
        return np.where(self.forward(inputs) >= 0, 1, -1)

    def fit(self, data, labels, max_epochs=100):
        """
        Fit the Perceptron to the training data using the original algorithm.

        Parameters
        ----------
        data : numpy.ndarray
            The training samples.
        labels : numpy.ndarray
            The target values.
        max_epochs : int, optional
            The maximum number of epochs. Defaults to 100.
        """
        
        print("\t\t---fit\n\n")
        
        for epoch in range(1, max_epochs+1):
            all_correct = True # Assume all predictions will be correct at the start of each epoch
            
            
            #self.epochs_.append(epoch)
            
            for inputs, label in zip(data, labels):
                prediction = self.predict(inputs)
                error = label - prediction

                #self.errors_.append(error)

                if error != 0:
                    all_correct = False # Set to False if any prediction is incorrect

                    # Update the weights and bias based on the error
                    update = self.learning_rate * error
                    self.weights += update * inputs
                    self.bias += update

            if all_correct:
                print(f"All predictions correct after {epoch + 1} epochs in training.")
                return

        print(f"Reached max_epochs ({max_epochs}).")
        print("\t\t---fit function ends\n\n")

    def fitGD(self, data, labels, max_epochs=1000, error_threshold=0.001):
        """
            If the error rate is 5% or less, stop gradient descent 
        
        

        Parameters
        ----------
        data : TYPE
            DESCRIPTION.
        labels : TYPE
            DESCRIPTION.
        max_epochs : TYPE, optional
            DESCRIPTION. The default is 100.
        error_threshold : TYPE, optional
            DESCRIPTION. The default is 0.001.

        Returns
        -------
        None.

        """
        print("\t\t--- fitGD\n\n")
        num_samples, num_features = data.shape
    
        for epoch in range(1, max_epochs+1):
            y_pred = self.forward(data)
    
            errors = labels - y_pred
            
            np.sum(errors)
            
            self.epochs_.append(epoch)
            
            
            
            
            # Mean Squared Error (MSE) Loss (variable used to assess convergence)
            mse_loss = (1 / num_samples) * np.sum(errors ** 2)
            self.loss_function_.append(mse_loss)
            
    
    
    
            #Gradient for weights
            dw = (-2 / num_samples) * np.dot(data.T, errors)
            # Gradient for bias
            db = (-2 / num_samples) * np.sum(errors)
    
            self.weights -= self.learning_rate * dw
            self.bias -= self.learning_rate * db
    
            # Check if predictions are correct
            all_correct = np.all(errors == 0)
    
            # Continually check convergence every 10 epochs
            if epoch % 10 == 0 or epoch == 1:
                print(f"Epoch {epoch}/{max_epochs}, MSE Loss: {mse_loss}")
    
            if all_correct or mse_loss < error_threshold:
                
                print(f"All predictions correct after {epoch + 1} epochs in fit_GD.")
                return
    
        print(f"Reached max_epochs ({max_epochs}).")
        
        print("\t\t--- fitGD ends\n\n")
        
        
    

#Calculate mean-squared error
def mse(y, y_hat):
    return np.mean((y - y_hat) ** 2)


if __name__ == "__main__":
    # Example usage with data and labels
    data = np.array([
    [0.5, -0.6],
    [1.0, -1.2],
    [-0.3, 0.4],
    [0.8, 0.2]
    ])
    labels = np.array([-1, 1, -1, 1])

    p = Perceptron(num_inputs=2, learning_rate=0.1)
    p.fit(data, labels, max_epochs=50)

    y = np.array([1,-1,1,-1])
    y_hat = np.array([1,1,-1,-1])
