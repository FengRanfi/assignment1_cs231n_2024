from builtins import range
import numpy as np
from random import shuffle
from past.builtins import xrange
import math

def softmax_loss_naive(W, X, y, reg):
    """
    Softmax loss function, naive implementation (with loops)

    Inputs have dimension D, there are C classes, and we operate on minibatches
    of N examples.

    Inputs:
    - W: A numpy array of shape (D, C) containing weights.
    - X: A numpy array of shape (N, D) containing a minibatch of data.
    - y: A numpy array of shape (N,) containing training labels; y[i] = c means
      that X[i] has label c, where 0 <= c < C.
    - reg: (float) regularization strength

    Returns a tuple of:
    - loss as single float
    - gradient with respect to weights W; an array of same shape as W
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using explicit loops.     #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score=X.dot(W)
    NNum=X.shape[0]
    CNum=W.shape[1]
    DNum=W.shape[0]
    maxList=[]
    maxArgList=[]
    for i in range(NNum):
        tmaxNum=0
        tmaxArgNum=0
        for j in range(CNum):
            if score[i,j]>tmaxNum:
                tmaxNum=score[i,j]
                tmaxArgNum=j
        maxList.append(tmaxNum)
        maxArgList.append(tmaxArgNum)
    sumList=[]
    for i in range(NNum):
        tSumNum=0
        for j in range(CNum):
            score[i,j]-=maxList[i]
            score[i,j]=math.exp(score[i,j])
            tSumNum+=score[i,j]
        sumList.append(tSumNum)
    for i in range(NNum):
        loss-=math.log(score[i,y[i]]/sumList[i])
    loss/=NNum
    for i in range(DNum):
        for j in range(CNum):
            loss+=reg*W[i][j]*W[i][j]

    XT=np.transpose(X)
    for i in range(NNum):
        for j in range(CNum):
            score[i,j]/=sumList[i]
    yOH= np.zeros_like(score)
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****
    for i in range(NNum):
        for j in range(CNum):
            if j==y[i]:
                yOH[i,j]=1
            else:
                yOH[i,j]=0
    dW=XT.dot(score-yOH)/NNum
    dW+=2*reg*W

    return loss, dW


def softmax_loss_vectorized(W, X, y, reg):
    """
    Softmax loss function, vectorized version.

    Inputs and outputs are the same as softmax_loss_naive.
    """
    # Initialize the loss and gradient to zero.
    loss = 0.0
    dW = np.zeros_like(W)

    #############################################################################
    # TODO: Compute the softmax loss and its gradient using no explicit loops.  #
    # Store the loss in loss and the gradient in dW. If you are not careful     #
    # here, it is easy to run into numeric instability. Don't forget the        #
    # regularization!                                                           #
    #############################################################################
    # *****START OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    score = X.dot(W)
    maxScore = np.max(score, axis=1)
    score2 = score - maxScore.reshape(X.shape[0], 1)
    eScore = np.exp(score2)
    eScoreSum = np.sum(eScore, axis=1)
    res = eScore / eScoreSum.reshape(X.shape[0], 1)
    res += 1e-15  # 加上一个很小的数，防止在求log的时候发生溢出。
    # print(res.shape,X.shape[0])
    # print(res[np.arange(X.shape[0]),y].shape)
    loss -= np.sum(np.log(res[np.arange(X.shape[0]), y]))
    loss /= X.shape[0]
    loss += reg * np.sum(W ** 2)

    yOH=np.zeros_like(score)
    yOH[np.arange(y.shape[0]),y]=1
    dW=np.transpose(X).dot(res-yOH)/X.shape[0]
    dW+=2*reg*W
    # *****END OF YOUR CODE (DO NOT DELETE/MODIFY THIS LINE)*****

    return loss, dW
