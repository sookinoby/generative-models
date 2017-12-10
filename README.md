# Generative Models using Apache MXNet

In our previous notebooks we have used deep learning to classify texts and images using convolution neural network. These models are generally refered as discriminative models. Discriminative models tries to create a decision boundary to classify a given input signal (data). Deep learning models can also be used to generate data based on the given input signal. These models are refered to as generative models. Generative models tries to learn the underlying distribution and can generate as well as classify input data. Generative models are powerful and can generate data similar to the input, providing insights to the phenomenon that generates the data.

Generative models can

Predict character/word prediction given previous character/word.
Synthesize new music or combine two genre of music to create entirely different music.
Synthesize new image from existing images

In general, Generative models can be used on any form of data to learn the underlying distribution and produce new data.



In this tutorial, we are going to build generative models Apache MXNet gluon API. We'll build up to a generative model that can predict the next character/word and build our own version of [swiftkey](https://blog.swiftkey.com/swiftkey-debuts-worlds-first-smartphone-keyboard-powered-by-neural-networks/). We learn about building blocks of Recurrent Neural Networt, implement a unrolled version of RNN, then move on to implementing LSTM(Long short term memory). We then move on to implement GAN (generative adversial neural network) using MXNet Gluon API, followed by simple neural network for style transfer.



This notebook expects you to have a basic understanding of recurrent neural network(RNN), activation units, gradient descent, and NumPy. 

By the end of the notebook, you will be able to:  

1. Understand generative models
2. Limitations of feed forward neural network
3. Understand the idea behind RNN and LSTM.
4. Installing mxnet with Gluon API.
5. Prepare datasets for training the neural network.
6. Implement a basic RNN using feed forward neural network
6. Implement a generative model to auto generate text using Gluon API
7. Implement generative adaptive neural network


# Generative models Martians vs Humans
Let us take a naive example to get better insights into the dataset. Suppose, we have given a task of classifying martian vs human based on their height in cm. Below is the sample data set

martian - 250,260,270,300,220,260,280,290,300,310
human - 160,170,180,190,175,140,180,210,140,200



If we train a discriminative model, the model will only learn a decision boundary (at height 200 cm). (manu figure 2 here). The model doesnt care about the underlying distribution that generates the data.

(manu figure 2 here)

In case of generative model, the model will learn the underlying distribution for martians (mean =274, std= 8.71) and human (mean=174, std=7.32). By extending this model we can generate new martians and humans, or a new interbreed species(humars). We can also use this model for classifying martians and humans.


#Limitations of feed forward neural network

Although feed forward neural network (including convolution neural network)have shown great accuracy in classifying sentences and text, its cannot store long term dependencies in its memory(hidden state). Convolution neural network can only remember spatial information for a small local neighbhour (siz of convolution kernels).  Let's consider a toy example which help us to understand long term dependencies.

```python
"<html>
<head>
<title>
RNN, Here I come.
 </title>
 </head> <body>Html is amazing, but I should not forget the end tag</body>
 </html>"
 ```

Suppose we need to train a model to generate prediction for auto complete feature for our latest super awesome text editor using deep learnig model. The model should remember long term dependiences like start tag <html> should be closed in the end. A convolution neural network does not have provision to remember long term context/information. A RNN can remember the context by passing along the hidden state.


# Inutiton behind RNN.

Manu figure (3)

Let us consider a problem of predicting the 4th character given the first 2 characters.  We can design a simple neural network as shown below ![Alt text](images/unRolled_rnn.png?raw=true "Unrolled RNN").


  This is basically feed forward network where the weights WI(green arrow), WH(Yello arrrow) are shared between some of the layers. This is an unrolled version of RNN  and this type of RNN are generally refered as many to one RNN, since N inputs (3 character) are used to predict one character. This can be designed in MxNet as follows

```python
class UnRolledRNN_Model(Block):
  # This is the initialisation of UnRolledRNN
    def __init__(self,vocab_size, num_embed, num_hidden,**kwargs):
        super(UnRolledRNN_Model, self).__init__(**kwargs)
        self.num_embed = num_embed
        self.vocab_size = vocab_size
        
        # use name_scope to give child Blocks appropriate names.
        # It also allows sharing Parameters between Blocks recursively.
        with self.name_scope():
            self.encoder = nn.Embedding(self.vocab_size, self.num_embed)
            self.dense1 = nn.Dense(num_hidden,activation='relu',flatten=True)
            self.dense2 = nn.Dense(num_hidden,activation='relu',flatten=True)
            self.dense3 = nn.Dense(vocab_size,flatten=True)

    # This is the forward pass of neural network
    def forward(self, inputs):
        emd = self.encoder(inputs)
        #print(emd.shape)
        #since the input is shape(batch_size,input(3 characters))
        # we need to extract 0th,1st,2nd character from each batch
        chararcter1 = emd[:,0,:]
        chararcter2 = emd[:,1,:]
        chararcter3 = emd[:,2,:]
        c1_hidden = self.dense1(chararcter1) # green arrow in diagram for character 1 (WI)
        c2_hidden = self.dense1(chararcter2) # green arrow in diagram for character 2 (WI)
        c3_hidden = self.dense1(chararcter3) # green arrow in diagram for character 3 (WI)
        c1_hidden_2 = self.dense2(c1_hidden)  # yellow arrow in diagram (WH)
        addition_result = F.add(c2_hidden,c1_hidden_2) # Total c1 + c2 
        addition_hidden = self.dense2(addition_result) # the yellow arrow (WH)
        addition_result_2 = F.add(addition_hidden,c3_hidden) # Total c2 + c3 
        final_output = self.dense3(addition_result_2)   # The red arrow in diagram (WO)  
        return final_output
  ```

There are other types are RNN models inculding the popular sequence to sequence RNN shown below ![Alt text](images/sequene_to_sequence.png?raw=true "Sequence to Sequence model").

Here N inputs (3 characters)  are mapped onto 3 outputs, this helps model to train faster as we loss at each time instant, so provides better feedback (back propagation) during model training. This model can be rolled back in and succinctly repsesented  ![Alt text](images/RNN.png?raw=true "RNN").

This representation also makes the math behind understanding easy to follow.

```python
hidden_state_at_t = (WI x input + WH x previous_hidden_state)
```


The are some limitation associated with basic RNN. For example, let us take a sentence like which contains 1000 words " I was born in france during world war.... So I can speak french". An simple RNN may not be able to reasons "being born in france" to "I can speak french" because there can be big distance between those two phrases. RNN doesnt provide provision to forget irrelvant context inbetween the phrases. Also training RNN over a long sequence can cause gradient in back propogation to vanish (when gradient is less one) or explode (gradient is bigger than 1) since back propogration basically mutiples the gradients along the graph in revese direction. Long short term memory (LSTM, a complex RNN unit) was proposed by German researchers Sepp Hochreiter and Juergen Schmidhuber as a solution to the vanishing / exploding gradient problem.

## Long short term memory (LSTM)

Long short term memory are type of RNN neural network which has two parameters, namely the hidden state and memory that are passed along the time step. Each unit of LSTM has small neural network that decides that amount of information that it needs to remember (memory) from previous time step. A LSTM neural network in abstract can be represented as below. A beautifully inllusrated in-depth description of LSTM can be found here [here ](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714) . ![Alt text](images/lstm.png?raw=true "RNN")



## Preparing your environment

If you're working in the AWS Cloud, you can save yourself the installation management by using a [Amazon Machine Image](https://aws.amazon.com/marketplace/pp/B01M0AXXQB#support), pre-configured for deep learning. This will enable you to skip steps 1-5 below.  

Note that if you are using a Conda environment, remember to install pip inside conda by typing 'conda install pip' after you activate an environment. This step will save you a lot of problems down the road.

Here's how to get set up: 

1. First, get [Anaconda](https://www.continuum.io/downloads), a package manager. It will help you to install dependent Python libraries with ease.
2. Next, install [scikit learn](http://scikit-learn.org/stable/install.html), a general-purpose scientific computing library. We'll use this to pre-process our data. You can install it with 'conda install scikit-learn'.
3. Then grab the Jupyter Notebook, with 'conda install jupyter notebook'.
4. And then, get [MXNet](https://github.com/apache/incubator-mxnet/releases), an open source deep learning library. The python notebook was tested on  0.12.0, you can with pip install mxnet==0.12.0


Here are the commands you need to type inside the anaconda environment (after its activation ):
1. conda install pip 
2. pip install opencv-python
3. conda install scikit-learn
4. conda install jupyter notebook
5. pip install mxnet==0.12.0


# Generative Models for characters

The notebook for this part of the tutorial can be found here 

I encourage you to download [the notebook](https://github.com/sookinoby/generative-models/blob/master/Test-rnn.ipynb) where we've created and run all this code, and play with it! Adjust the hyperparameters and experiment with different approaches to neural network architecture.

## Preparing the DataSet

In order to learn about any deep neural network, we need data. For this notebook, we'll use a text literary work of [Friedrich Nietzsche](https://en.wikipedia.org/wiki/Friedrich_Nietzsche). You can download the data set [here](https://s3.amazonaws.com/text-datasets/nietzsche.txt). You are free to use any other dataset or try something from [here](https://cs.stanford.edu/people/karpathy/char-rnn/)

The data set nietzsche.txt consists of 600901 characters, with 86 unique character. We need to encode the complete text as sequence of numbers. This can be done as follows

```python
chars = sorted(list(set(text)))
#maps character to unique index e.g. {a:1,b:2....}
char_indices = dict((c, i) for i, c in enumerate(chars))
#maps indices to character (1:a,2:b ....)
indices_char = dict((i, c) for i, c in enumerate(chars))
#convert the entire text into sequence
idx = [char_indices[c] for c in text]
```

### preparing dataset for Un-rolled RNN
 We need to convert the dataset in such a way that we can split inputs into sequence of `3 character` (input) and 4th character being the output.

 ```python
 #input for neural network( our basic rnn has 3 inputs, n samples)
cs=3
c1_dat = [idx[i] for i in range(0, len(idx)-1-cs, cs)]
c2_dat = [idx[i+1] for i in range(0, len(idx)-1-cs, cs)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-1-cs, cs)]
#the output of rnn network (single vector)
c4_dat = [idx[i+3] for i in range(0, len(idx)-1-cs, cs)]
#stacking the inputs to form (3 input features )
x1 = np.stack(c1_dat[:-2])
x2 = np.stack(c2_dat[:-2])
x3 = np.stack(c3_dat[:-2])

# Concatenate to form the input training set
col_concat = np.array([x1,x2,x3])
t_col_concat = col_concat.T

```
We also the batchify the training set in batches of 32, so each training instance is of shape 32 X 3

```python
#Set the batchsize as 32, so input is of form 32 X 3
#output is 32 X 1
batch_size = 32
def get_batch(source,label_data, i,batch_size=32):
    bb_size = min(batch_size, source.shape[0] - 1 - i)
    data = source[i : i + bb_size]
    target = label_data[i: i + bb_size]
    #print(target.shape)
    return data, target.reshape((-1,))
```

### preparing dataset for gluon RNN

This is very similar to preparing dataset for un Rolled RNN expect for ordering. The dataset should be ordered in the shape (number of example X batch_size). For example let consider sample dataset set in the format . Need to continue


