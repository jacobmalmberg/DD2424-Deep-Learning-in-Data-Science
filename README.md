# DD2424-Deep-Learning-in-Data-Science
Completed assignments for course DD2424 Deep Learning in Data Science. Course information can be found at https://www.kth.se/student/kurser/kurs/DD2424?l=en. 

Course contents

* Learning of representations from raw data: images and text
* Principles of supervised learning
* Elements for different methods for deep learning: convolutional networks and recurrent networks
* Theoretical knowledge of and practical experience of training networks for deep learning including optimisation using stochastic gradient descent
* New progress in methods for deep learning
* Analysis of models and representations
* Transferred learning with representations for deep learning
* Application examples of deep learning for learning of representations and recognition


## Assignment 1.

The instructions can be found [here](https://github.com/jacobmalmberg/DD2424-Deep-Learning-in-Data-Science/blob/master/Assignment%201/Assignment1.pdf) and the full report can be found [here](https://github.com/jacobmalmberg/DD2424-Deep-Learning-in-Data-Science/blob/master/Assignment%201/DD2424_Ass1.pdf).

The first assignment involves training a one layer network with multiple outputs to classify images from the CIFAR-10 dataset. Two variants were implemented, one using cross-entropy loss and one using SVM multi-class loss. The network was optimized to achieve the best performance using decaying learning rate and shuffling of the data.

## Assignment 2.

The instructions can be found [here](https://github.com/jacobmalmberg/DD2424-Deep-Learning-in-Data-Science/blob/master/Assignment%202/Assignment2.pdf) and the full report can be found [here](https://github.com/jacobmalmberg/DD2424-Deep-Learning-in-Data-Science/blob/master/Assignment%202/DD2424_Ass2_basic-1.pdf).

The second assignment involves training a two layer network with multiple outputs to classify images from the CIFAR-10 dataset. The network was trained using mini-batch gradient descent applied to a cost function that computes the cross-entropy loss of the classifier applied to the labelled training data and an L2 regularization term on the weight matrix. Cyclic learning rate was used. The network was optimized by using additional hidden nodes as well as dropout.

## Assignment 3.

The instructions can be found [here](https://github.com/jacobmalmberg/DD2424-Deep-Learning-in-Data-Science/blob/master/Assignment%203/Assignment3.pdf) and the full report can be found [here](https://github.com/jacobmalmberg/DD2424-Deep-Learning-in-Data-Science/blob/master/Assignment%203/DD2424_Ass3.pdf).

The third assignment involves implementing a two layer ConvNet to predict the language of a surname from its' spelling from scratch. As the dataset was unbalanced, sampling of the training data during back-prop training was done. As ConvNet's are computationally expensive, extra measures such as sparse matrices were taken to minimize running time.

## Assignment 4.

The instructions can be found [here](https://github.com/jacobmalmberg/DD2424-Deep-Learning-in-Data-Science/blob/master/Assignment%204/Assignment4.pdf) and the full report can be found [here](https://github.com/jacobmalmberg/DD2424-Deep-Learning-in-Data-Science/blob/master/Assignment%204/DD2424_Ass4.pdf).

The fourth assignment involves implementing an RNN to synthesize English text character
by character. AdaGrad was used to train the network. Text was synthethized from a Harry Potter book as well as from Donald Trump's twitter account.

### Sample generated text

* ’hating Obamahine. I wall the worst, PAMCNELDED, was heve bad ald
correc’
