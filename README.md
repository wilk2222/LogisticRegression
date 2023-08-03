# LogisticRegression 
---
Implementation of LogisticRegression 📈 with regularisation parameter λ.

Basic implementation of **Binary Logistic Regression** from scratch using *Python* and *numpy*. The binary logistic regression model can be thought of as a simple neural network with just one output node with a sigmoid activation function and no hidden layers.

##**Mathematical equations used in the code**

### **Linear Combination (weighted sum)**
If you have a feature matrix X with $m$ examples and $n$ features, and a weight vector $theta$ of size $n$, then the linear combination $z$ is computed as :

$$z = X \theta$$

### **Sigmoid Function**
This function is used to map the linear combination to a range between 0 and 1, which can be interpreted as a probability.

$$h = \frac{1}{1 + e^{-z}}$$

### **Binary Cross-Entropy Loss**
The Loss function used for binary classification in logistic regression is binary cross-entropy. Here $y$ is the true label and $h$ is the predicted probability.

$$L(y, h) = -[y \log(h) + (1 - y) \log(1 - h)]$$

### **Regularization Term**
The regularization term is a penalty added to the loss function to reduce the magnitude of the weights and therefore reduce overfitting. Here, $theta$ is the weight vector and $lambda$ is the regularization parameter.

$$R(\theta) = \frac{\lambda}{2m} \sum \theta^2$$

The total loss function including the regularization term is then:

$$J(\theta) = L(y, h) + R(\theta)$$

### **Gradient Calculation**
The gradients are the deratives of the loss function with respect to the weights. For logistic regression with regularization, it denoted by:

$$\text{gradient} = \frac{1}{m} X^T (h - y) + \frac{\lambda}{m} \theta$$

### **Weight Update Rule**
The weights are udpated using gradient descent. The learning rate is denoted by $alpha$.

$$\theta = \theta - \alpha \cdot \text{gradient}$$

---




