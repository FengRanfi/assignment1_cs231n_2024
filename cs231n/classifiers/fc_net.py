from builtins import range
from builtins import object
import numpy as np

from ..layers import *
from ..layer_utils import *


class TwoLayerNet(object):
    """
    A two-layer fully-connected neural network with ReLU nonlinearity and
    softmax loss that uses a modular layer design. We assume an input dimension
    of D, a hidden dimension of H, and perform classification over C classes.

    The architecure should be affine - relu - affine - softmax.

    Note that this class does not implement gradient descent; instead, it
    will interact with a separate Solver object that is responsible for running
    optimization.

    The learnable parameters of the model are stored in the dictionary
    self.params that maps parameter names to numpy arrays.
    """

    def __init__(
        self,
        input_dim=3 * 32 * 32,
        hidden_dim=100,
        num_classes=10,
        weight_scale=1e-3,
        reg=0.0,
    ):
        """
        Initialize a new network.

        Inputs:
        - input_dim: An integer giving the size of the input
        - hidden_dim: An integer giving the size of the hidden layer
        - num_classes: An integer giving the number of classes to classify
        - weight_scale: Scalar giving the standard deviation for random
          initialization of the weights.
        - reg: Scalar giving L2 regularization strength.
        """
        self.params = {}
        self.reg = reg
        self.input_dim=input_dim
        ############################################################################
        # TODO: Initialize the weights and biases of the two-layer net. Weights    #
        # should be initialized from a Gaussian centered at 0.0 with               #
        # standard deviation equal to weight_scale, and biases should be           #
        # initialized to zero. All weights and biases should be stored in the      #
        # dictionary self.params, with first layer weights                         #
        # and biases using the keys 'W1' and 'b1' and second layer                 #
        # weights and biases using the keys 'W2' and 'b2'.                         #
        # 初始化两层网络的权重和偏置。权重应从以0.0为中心的高斯开始初始化，标准差等于
        # weight_scale，偏差应初始化为零。所有权重和偏差都应存储在字典
        # self.params
        # 中，第一层权重和偏差使用键'W1'和'b1'，第二层权重和偏差使用键'W2'和'b2'。
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        self.params["W1"]=np.random.normal(0,weight_scale,size=(input_dim,hidden_dim))           #ID,HD
        self.params["b1"]=np.zeros((1,hidden_dim))
        self.params["W2"]=np.random.normal(0,weight_scale,size=(hidden_dim,num_classes))
        self.params["b2"]= np.zeros((1,num_classes))
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

    def loss(self, X, y=None):
        """
        Compute loss and gradient for a minibatch of data.

        Inputs:
        - X: Array of input data of shape (N, d_1, ..., d_k)
        - y: Array of labels, of shape (N,). y[i] gives the label for X[i].

        Returns:
        If y is None, then run a test-time forward pass of the model and return:
        - scores: Array of shape (N, C) giving classification scores, where
          scores[i, c] is the classification score for X[i] and class c.

        If y is not None, then run a training-time forward and backward pass and
        return a tuple of:
        - loss: Scalar value giving the loss
        - grads: Dictionary with the same keys as self.params, mapping parameter
          names to gradients of the loss with respect to those parameters.
        """
        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the two-layer net, computing the    #
        # class scores for X and storing them in the scores variable.              #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        X=X.reshape(X.shape[0],self.input_dim)
        W1=self.params["W1"]
        b1=self.params["b1"]
        W2=self.params["W2"]
        b2=self.params["b2"]
        y1=X.dot(W1)+b1
        y1[y1<0]=0  #relu
        y2=y1.dot(W2)+b2
        y2_max=np.max(y2,axis=1)
        y2_n=y2-y2_max.reshape(X.shape[0],1)
        y2_ne=np.exp(y2_n)
        y2_neSum=np.sum(y2_ne,axis=1)
        scores2=y2_ne/y2_neSum.reshape(X.shape[0],1)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################
        scores=y2
        # If y is None then we are in test mode so just return scores
        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the two-layer net. Store the loss  #
        # in the loss variable and gradients in the grads dictionary. Compute data #
        # loss using softmax, and make sure that grads[k] holds the gradients for  #
        # self.params[k]. Don't forget to add L2 regularization!                   #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

        loss=-np.sum(np.log(scores2[np.arange(X.shape[0]),y]))/X.shape[0]+(np.sum(self.reg*W1**2)+np.sum(self.reg*W2**2))/2
        ty=np.zeros((X.shape[0],W2.shape[1]))
        ty[np.arange(X.shape[0]),y]=1
        Loss_score=scores2-ty
        grads["W2"]=np.transpose(y1).dot(Loss_score)/X.shape[0]+self.reg*W2
        grads["b2"]=np.mean(Loss_score,axis=0)
        y1_static=np.copy(y1)
        y1_static[y1>0]=1
        pre_grad=Loss_score.dot(np.transpose(W2))*y1_static
        grads["W1"]=np.transpose(X).dot(pre_grad)/X.shape[0]+self.reg*W1
        grads["b1"]=np.mean(pre_grad,axis=0)
        # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads
