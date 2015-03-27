# Stanford-MachineLearning
[Stanford Coursera Machine Learning](https://www.coursera.org/course/ml)  
Andrew Ng  

#MATLAB
![](/img/logo.png)  
* The heart of MATLAB is matrix.  
* Default data type is double.
* Lambda: `g = arrayfun(@(x) 1/(1+exp(-x)), z);`.
* Mathematical operations use index starting from 1. And `X(1, :)` is different from `X(1)`.
* `A(:)` is used for matrix unrolling to vector.
* `theta'*theta` is different from `theta*theta'`; thus `theta .^2` is preferred.
* `dpquit` to quit the debug mode.
* `X(2:end, :)`: use `end` for slicing.
* Cell array is indexed by `A{1}`.
* `~` to skip a return value: `[U, S, ~] = svd(Sigma)`.
* Matrix multiplication orders depend on whether the data point is a col vector or row vector. 
* For loop: `for epsilon = min(pval):stepsize:max(pval)`

# Index
## Linear Regression with Multiple Variables
1. Cost function for one var
1. Gradient descent for one var
1. Feature normalization
1. Cost function for multi-var
1. Gradient descent for multi-var
1. Normal Equations 

<img src="http://latex.codecogs.com/gif.latex?J(\theta)=\frac{1}{2m}\sum_{i=1}^{m}{\big(h_\theta(x^{(i)})-y^{(i)}\big)^2}" />  
<img src="http://latex.codecogs.com/gif.latex?h_\theta(x)=\theta^Tx" />  
<img src="http://latex.codecogs.com/gif.latex?\theta_j:=\theta_j-\alpha\frac{1}{m}\sum_{i=1}^{m}{\big(h_\theta(x^{(i)})-y^{(i)}\big)x_j^{(i)}}" />  

## Regularization - Logistic Regression 
1. Sigmoid function
1. Cost function for logistic regression (LR)
1. Gradient descent for LR
1. Predict function (hypothesis)
1. Cost function for regularized LR 
1. Gradient descent for regularized LR 

<img src="http://latex.codecogs.com/gif.latex?h_\theta(x)=g(\theta^Tx)" />  
<img src="http://latex.codecogs.com/gif.latex?g(z)=\frac{1}{1+e^{-z}}" />  
<img src="http://latex.codecogs.com/gif.latex?J(\theta)=\frac{1}{m}\sum_{i=1}^{m}{\big[-y^{(i)}\log{(h_\theta(x^{(i)}))}-(1-y^{(i)})\log{(1-h_\theta(x^{(i)}))}\big]}+\frac{\lambda}{2m}\sum_{j=1}^n{\theta_j^2}" />  
<img src="http://latex.codecogs.com/gif.latex?\frac{\partial J(\theta)}{\partial \theta_j} = \Bigg(\frac{1}{m}\sum_{i=1}^m{\big(h_\theta(x^{(i)})-y^{(i)}\big)x_j^{(i)}}\Bigg)+\frac{\lambda}{m}\theta_j" />  

## Neural Networks: Representation
1. Regularized Logistic Regression 
1. One-vs-all classifier training 
1. One-vs-all classifier prediction 
1. [Neural Network predict function](https://github.com/idf/Stanford-MachineLearning/blob/develop/mlclass-ex3-008%2Fmlclass-ex3%2Fpredict.m)

<img src="http://latex.codecogs.com/gif.latex?J(\theta)=\frac{1}{m}\sum{\big[-y\circ\log{(h_\theta(X))}-(1-y)\circ\log{(1-h_\theta(X))}\big]}+\frac{\lambda}{2m}\sum{\theta\circ\theta}" />  
<img src="http://latex.codecogs.com/gif.latex?\frac{\partial J(\theta)}{\partial \theta} = \frac{1}{m}X^T\big(h_\theta(X)-y\big)+\frac{\lambda}{m}\theta" />  

## Neural Networks: Learning
1. Feedforward and cost function 
1. Regularized cost function 
1. Sigmoid gradient
1. Neural Net gradient function (Backpropagation) 
1. Regularized gradient

<img src="http://latex.codecogs.com/gif.latex?g'(z)=g(z)(1-g(z))" />  
<img src="http://latex.codecogs.com/gif.latex?J(\theta)=\frac{1}{m}\sum_{i=1}^{m}\sum_{k=1}^K{{\big[-y_k^{(i)}\log{(h_\theta(x^{(i)}))_k}-(1-y_k^{(i)})\log{(1-h_\theta(x^{(i)}))_k}\big]}}+\frac{\lambda}{2m}\sum_{l}{\sum_{j\in (l+1)}{\sum_{k\in l}{(\Theta_{j,k}^{(l)})^2}}}" />  
<img src="http://latex.codecogs.com/gif.latex?\delta_k^{(out)}=a_k^{(out)}-y_k" />  
<img src="http://latex.codecogs.com/gif.latex?\delta^{(l)}= (\Theta^{(l)})^T\delta^{(l+1)}\circ g'(z^{(l)})" />  
<img src="http://latex.codecogs.com/gif.latex?\Delta^{(l)}:=\Delta^{(l)}+\delta^{(l+1)}(a^{(l)})^T" />  
<img src="http://latex.codecogs.com/gif.latex?\frac{\partial}{\partial \Theta_{ij}^{(l)}}J(\Theta)=D_{ij}^{(l)}=\frac{1}{m}\Delta_{ij}^{(l)}+\frac{\lambda}{m}\Theta_{ij}^{(l)}" />  

## Regularized Linear Regression and Bias/Variance
1. Regularized LR, cost function (review)
1. Regularized LR, gradient (review)
1. Learning Curve - Bias-Variance trade-off
1. Polynomial feature mapping 
1. Cross validation curve - (select lambda)

<img src="http://latex.codecogs.com/gif.latex?h_\theta(x)=\theta_0+\theta_1 x_1+...+\theta_p x_p" />, where `x_i = normalize(x .^ i)`
## Support Vector Machines 
1. Gaussian Kernel
1. Parameters (C, sigma)
1. Email preprocessing
1. Email feature extraction 

<img src="http://latex.codecogs.com/gif.latex?\operatornamewithlimits{min}_\theta C\sum_{i=1}^{m}{\big[y^{(i)}cost_1{(\theta^Tx^{(i)})}+(1-y^{(i)})cost_0{(\theta^Tx^{(i)})}\big]}+\frac{1}{2}\sum_{j=1}^n{\theta_j^2}" />  
<img src="http://latex.codecogs.com/gif.latex?K_{gaussian}(x^{(i)}, x^{(j)})=\exp{\Bigg(-\frac{||x^{(i)}-x^{(j)}||^2}{2\sigma^2}\Bigg)}" />  
<img src="http://latex.codecogs.com/gif.latex?\operatornamewithlimits{min}_\theta C\sum_{i=1}^{m}{\big[y^{(i)}cost_1{(\theta^Tf^{(i)})}+(1-y^{(i)})cost_0{(\theta^Tf^{(i)})}\big]}+\frac{1}{2}\sum_{j=1}^n{\theta_j^2}" />  
<img src="http://latex.codecogs.com/gif.latex?f_k^{(i)} = K(x^{(i)}, l^{(k)})" />  

## K-Means Clustering and PCA
1. Find closest centroids
1. Compute centroid means
1. PCA
1. Project data
1. Recover data

<img src="http://latex.codecogs.com/gif.latex?c^{(i)}:= \operatornamewithlimits{argmin}_{j} ||x^{(i)}-\mu_j||^2" />   
<img src="http://latex.codecogs.com/gif.latex?\mu_k:=\frac{1}{C_k}\sum_{i \in C_k}{x^{(i)}}" />  
<img src="http://latex.codecogs.com/gif.latex?\Sigma=\frac{1}{m}X^TX" />  

## Anomaly Detection and Recommender Systems
1. Estimate Gaussian parameters
1. Select threshold
1. Collaborative Filtering cost
1. Collaborative Filtering gradient
1. Regularized cost
1. Gradient with regularization

<img src="http://latex.codecogs.com/gif.latex?p(x;\mu,\sigma^2)=\frac{1}{\sqrt{2\pi\sigma^2}}\exp{\Big(-\frac{(x-\mu)^2}{2\sigma^2}\Big)}" />  
<img src="http://latex.codecogs.com/gif.latex?J(x^{(i)},...,x^{(n_m)},\theta^{1},...,\theta^{(n_u)})=\frac{1}{2}\sum_{(i,j):r(i,j)=1}{((\theta^{(j)})^Tx^{(i)}-y^{(i,j)})^2}+\frac{\lambda}{2}\sum_{j=1}^{n_u}{\sum_{k=1}^n{(\theta_k^{(j)}})^2}+\frac{\lambda}{2}\sum_{j=1}^{n_m}{\sum_{k=1}^n{(x_k^{(j)}})^2}" />  
<img src="http://latex.codecogs.com/gif.latex?\frac{\partial J}{\partial x_k^{(i)}}=\sum_{j:r(i,j)=1}{\big((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\big)\theta_k^{(j)}}+\lambda x_k^{(i)}" />  
<img src="http://latex.codecogs.com/gif.latex?\frac{\partial J}{\partial \theta_k^{(j)}}=\sum_{i:r(i,j)=1}{\big((\theta^{(j)})^Tx^{(i)}-y^{(i,j)}\big)x_k^{(i)}}+\lambda \theta_k^{(j)}" />  
