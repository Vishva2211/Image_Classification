# Image_Classification
1)Load the dataset in batches (16,32,64 or 128 images per batch).
2)Convert all images from RGB to Grayscale.
3)Define a filter of shape (2,5,5), where 2 indicates the number of filters.
4)Pre-process the images…normalize, gaussian blur, etc.
5)convolve the output images using the filter 1)the size of output after convolution will be n-f+1 where f is the filter size
6)apply normalization to the output and apply relu function (which considers positive numbers as it is and negative numbers as zero).
7)Apply max pooling …size will be ….(input shape-size of pool kernel+1)/stride
8)Now reshape the output to a linear column matrix.
9)Now we have an output of size (number of elements in a single linear column matrix) x (number of images per batch).
10)Now , send the output of conv as an input to fully connected neural network layer.
11)initiate the parameters (W1,b1,W2,b2,W3,b3) of sizes((100,size of linear column matrix),(100,batch_size),(50,100),(50,batch_size),(2,50),(2,batch_size))
12)Forward propagate the output …
Z1 = W1.dot(X) + b1
A1 = ReLU(Z1)
ZH1 = W2.dot(A1) + b2
AH1 = ReLU(ZH1)
Z2 = W3.dot(AH1) + b3
A2 = softmax(Z2)
13)ReLU returns the same number if its positive and zero if the number is negative
14)Softmax function takes the exponent of an element and divides itb bt the sum of exponents of all other elements in the matrix
15)We use log loss in our case(binary cross-entropy) (-1/m)*Sum(1 to m)(yi.log(Yi)) + (1-yi).log(1-Yi)) 16)Now differentiate this loss function wrt all the parameters and we get,
dZ2 = A2 - final_Y
dW3 = 1 / m * dZ2.dot(AH1.T)
db3 = 1 / m * numpy.sum(dZ2)
dZH1 = W3.T.dot(dZ2) * derivative_activation(ZH1)
dW2 = 1 / m * dZH1.dot(A1.T)
db2 = 1 / m * numpy.sum(dZH1)
dZ1 = W2.T.dot(dZH1) * derivative_activation(Z1)
dW1 = 1 / m * dZ1.dot(X.T)
db1 = 1 / m * np.sum(dZ1)
m is the number of images per batch
here final_Y is the one hot encoded version of Y
17)Now update the parameters with some learning rate alpha
W1 = W1 - alpha * dW1
b1 = b1 - alpha * db1
W2 = W2 - alpha * dW2
b2 = b2 - alpha * db2
W3 = W3 - alpha * dW3
b3 = b3 - alpha * db3
18)run this for some number of iterations like 1000
19)then finally,find the predictions using argmax function.
20)Then to find accuracy,it’s the number of correct predictions/size of labels.
21)Now save the parameters.
