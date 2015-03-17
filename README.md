# Stanford-MachineLearning
[Stanford Coursera Machine Learning](https://www.coursera.org/course/ml)  
Andrew Ng  

#MATLAB
![](/img/logo.png)  
* The heart of MATLAB calculation is matrix.  
* `g = arrayfun(@(x) 1/(1+exp(-x)), z);`
* Mathematical operations use index starting from 1. And `X(1, :)` is different from `X(1)`  
* `A(:)` is used for matrix unrolling to vector
* `theta'*theta` is different from `theta*theta'`; thus `theta .^2 is preferred`
* `dpquit` to quit the debug mode 
* `X(2:end, :)`, use `end` for slicing 
* cell array is indexed by `A{1}`

# Index
## Linear Regression with Multiple Variables
1. Cost function for one var
1. Gradient descent for one var
1. Feature normalization
1. Cost function for multi-var
1. Gradient descent for multi-var
1. Normal Equations 

## Regularization - Logistic Regression 
1. Sigmoid function
1. Cost function for logistic regression (LR)
1. Gradient descent for LR
1. Predict function (hypothesis)
1. Cost function for regularized LR 
1. Gradient descent for regularized LR 

## Neural Networks: Representation
1. Regularized Logistic Regression 
1. One-vs-all classifier training 
1. One-vs-all classifier prediction 
1. Neural Network predict function 

## Neural Networks: Learning
1. Feedforward and cost function 
1. Regularized cost function 
1. Sigmoid gradient
1. Neural Net gradient function (Backpropagation) 
1. Regularized gradient

## Regularized Linear Regression and Bias/Variance
1. Regularized Linear Regression Cost Function (review)
1. Regularized Linear Regression Gradient (review)
1. Learning Curve - bias-variance tradeoff
1. Polynomial Feature Mapping 
1. Cross Validation Curve - (select lambda)

## Support Vector Machines 
1. Gaussian Kernel
1. Parameters (C, sigma)
1. Email Preprocessing
1. Email Feature Extraction 

## K-Means Clustering and PCA
1. Find Closest Centroids
1. Compute Centroid Means
1. PCA
1. Project Data
1. Recover Data