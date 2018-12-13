from builtins import object
import numpy as np

from cs231n.layers import *
from cs231n.fast_layers import *
from cs231n.layer_utils import *


class ThreeLayerConvNet(object):
    """
    A three-layer convolutional network with the following architecture:

    conv - relu - 2x2 max pool - affine - relu - affine - softmax

    The network operates on minibatches of data that have shape (N, C, H, W)
    consisting of N images, each with height H and width W and with C input
    channels.
    """

    def __init__(self, input_dim=(3, 32, 32), num_filters=32, filter_size=7,
                 hidden_dim=100, num_classes=10, weight_scale=1e-3, reg=0.0,
                 dtype=np.float32):
        """
        Initialize a new network.

        Inputs:
        - input_dim: Tuple (C, H, W) giving size of input data   输入参数的尺寸
        - num_filters: Number of filters to use in the convolutional layer 卷积核数目
        - filter_size: Width/height of filters to use in the convolutional layer 卷积核尺寸
        - hidden_dim: Number of units to use in the fully-connected hidden layer 全连接层的神经元数
        - num_classes: Number of scores to produce from the final affine layer. 分类数
        - weight_scale: Scalar giving standard deviation for random initialization
          of weights. 随机初始化的标准差
        - reg: Scalar giving L2 regularization strength 正则化强度
        - dtype: numpy datatype to use for computation. 计算的数据类型
        """
        self.params = {}
        self.reg = reg
        self.dtype = dtype

        ############################################################################
        # TODO: Initialize weights and biases for the three-layer convolutional    #
        # network. Weights should be initialized from a Gaussian centered at 0.0   #
        # with standard deviation equal to weight_scale; biases should be          #
        # initialized to zero. All weights and biases should be stored in the      #
        #  dictionary self.params. Store weights and biases for the convolutional  #
        # layer using the keys 'W1' and 'b1'; use keys 'W2' and 'b2' for the       #
        # weights and biases of the hidden affine layer, and keys 'W3' and 'b3'    #
        # for the weights and biases of the output affine layer.                   #
        #                                                                          #
        # IMPORTANT: For this assignment, you can assume that the padding          #
        # and stride of the first convolutional layer are chosen so that           #
        # **the width and height of the input are preserved**. Take a look at      #
        # the start of the loss() function to see how that happens.                #                           
        ############################################################################
        #初始化w1，b1，w2，b2，放在self.params中,假设卷积后没有降维
        C,H,W=input_dim
        self.params['W1']=weight_scale*np.random.randn(num_filters,C,filter_size,filter_size)
        #卷积核数量，深度，长，宽
        self.params['b1']=np.zeros(1,num_filters)
        self.params['W2']=weight_scale*np.random.randn(num_filters*H*W/4,hidden_dim)
        #前面有假设是卷积后没降维，所以是num_filters*H*W，pooling后再除以4
        self.params['b2']=np.zeros(1,hidden_dim)
        self.params['W3']=weight_scale*np.random.randn(hidden_dim,num_classes)
        self.params['b3']=np.zeros(1,num_classes)
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        for k, v in self.params.items():
            self.params[k] = v.astype(dtype)


    def loss(self, X, y=None):
        """
        Evaluate loss and gradient for the three-layer convolutional network.
        评估这个三层cnn的损失和梯度

        Input / output: Same API as TwoLayerNet in fc_net.py.
        """
        W1, b1 = self.params['W1'], self.params['b1']
        W2, b2 = self.params['W2'], self.params['b2']
        W3, b3 = self.params['W3'], self.params['b3']

        # pass conv_param to the forward pass for the convolutional layer
        # Padding and stride chosen to preserve the input spatial size
        filter_size = W1.shape[2]
        conv_param = {'stride': 1, 'pad': (filter_size - 1) // 2}
        #为了保证不降维，padding就这么算的

        # pass pool_param to the forward pass for the max-pooling layer
        pool_param = {'pool_height': 2, 'pool_width': 2, 'stride': 2}

        scores = None
        ############################################################################
        # TODO: Implement the forward pass for the three-layer convolutional net,  #
        # computing the class scores for X and storing them in the scores          #
        # variable.                                                                #
        #                                                                          #
        # Remember you can use the functions defined in cs231n/fast_layers.py and  #
        # cs231n/layer_utils.py in your implementation (already imported).         #
        ############################################################################
        #执行前向传播，计算loss
        a1,cache1=conv_relu_pool_forward(X,W1,b1,conv_param,pool_param)
        #卷积，relu，pool,chace里·貌似是层的信息
        a2,cache2=affine_relu_forward(a1,W2,b2)
        #全连接，rulu
        scores,cache3=affine_forward(a2,W3,b3)
        #全连接,cache3是a2,W3,b3
        
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        if y is None:
            return scores

        loss, grads = 0, {}
        ############################################################################
        # TODO: Implement the backward pass for the three-layer convolutional net, #
        # storing the loss and gradients in the loss and grads variables. Compute  #
        # data loss using softmax, and make sure that grads[k] holds the gradients #
        # for self.params[k]. Don't forget to add L2 regularization!               #
        #                                                                          #
        # NOTE: To ensure that your implementation matches ours and you pass the   #
        # automated tests, make sure that your L2 regularization includes a factor #
        # of 0.5 to simplify the expression for the gradient.                      #
        ############################################################################
        #执行反向传播
        #计算loss
        reg=self.reg   
        
        
        #data_loss, dscores = softmax_loss(scores, y)
        
        
        N=X.shape[0]
        scores=scores-np.reshape(np.max(scores,axis=1),(N,-1))
        p=np.exp(scores)/np.reshape(np.sum(np.exp(scores),axis=1),(N,-1))
        loss=-sum(np.log(p[np.arange(N),y]))/N
        loss+=0.5*reg*np.sum(W1*W1)+0.5*reg*np.sum(W2*W2)+0.5*reg*np.sum(W3*W3)
        #反向传播
        
        
        dscores = p
        dscores[range(N),y]-=1.0
        
        da2, dw3, db3=affine_backward(dscores, cache3)
        #cache3是a2,W3,b3,计算梯度时需要
        #得到的是
        dw3+=self.reg*W3
        grads['W3']=dw3
        grads['b3']=db3
        da1, dw2, db2=affine_relu_backward(da2, cache2)
        #da2为上游传下来的梯度，cache2是a1,W2,b2
        dw2+=self.reg*W2
        grads['W2']=dw2
        grads['b2']=db2
        dx, dw1, db1=conv_relu_pool_backward(da1, cache1)
        #cache1是X,W1,b1
        dw1+=self.reg*W1
        grads['W1']=dw1
        grads['b1']=db1       
        ############################################################################
        #                             END OF YOUR CODE                             #
        ############################################################################

        return loss, grads






















