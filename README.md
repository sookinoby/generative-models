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

In order to learn about any deep neural network, we need data. For this notebook, we'll use a text literary work of [Friedrich Nietzsche](https://en.wikipedia.org/wiki/Friedrich_Nietzsche). You can download the data set [here](https://s3.amazonaws.com/text-datasets/nietzsche.txt). You are free to use any other dataset including your own chat history or try something from [here](https://cs.stanford.edu/people/karpathy/char-rnn/)

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

This is very similar to preparing dataset for un Rolled RNN expect for ordering. The dataset should be ordered in the shape (number of example X batch_size). For example let consider sample dataset set in the format below.
![Alt text](images/batch_reshape.png?raw=true "batch reshape")
The input sequence is converted to batch of size 3, and then into 2 separate input sequence of length. By performing such a transformation, it is very easy to generate arbitary length input sequence for trainign.

### Designing RNN in Gluon

We define a class which allows us to create two type of RNN namely GRU (Gated Recurrent Unit) and LSTM. Below the python snippet

```python
# Class to create model objects.
class GluonRNNModel(gluon.Block):
    """A model with an encoder, recurrent layer, and a decoder."""

    def __init__(self, mode, vocab_size, num_embed, num_hidden,
                 num_layers, dropout=0.5, **kwargs):
        super(GluonRNNModel, self).__init__(**kwargs)
        with self.name_scope():
            self.drop = nn.Dropout(dropout)
            self.encoder = nn.Embedding(vocab_size, num_embed,
                                        weight_initializer = mx.init.Uniform(0.1))
               
            if mode == 'lstm':
                self.rnn = rnn.LSTM(num_hidden, num_layers, dropout=dropout,
                                    input_size=num_embed)
            elif mode == 'gru':
                self.rnn = rnn.GRU(num_hidden, num_layers, dropout=dropout,
                                   input_size=num_embed)
            else:
                self.rnn = rnn.RNN(num_hidden, num_layers, activation='relu', dropout=dropout,
                                   input_size=num_embed)
            self.decoder = nn.Dense(vocab_size, in_units = num_hidden)
            self.num_hidden = num_hidden
    #define the forward pass of the neural network
    def forward(self, inputs, hidden):
        emb = self.drop(self.encoder(inputs))
        output, hidden = self.rnn(emb, hidden)
        output = self.drop(output)
        decoded = self.decoder(output.reshape((-1, self.num_hidden)))
        return decoded, hidden
    #Initial state of netork
    def begin_state(self, *args, **kwargs):
        return self.rnn.begin_state(*args, **kwargs)
```
The constructor of class creates few neural units that will be used in our forward pass. The forward pass is the method that will be called during our training to generate the loss.
Then We first create an [embedding layer](https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.nn.Embedding) for the input character. You can look at the [previous blog](https://www.oreilly.com/ideas/sentiment-analysis-with-apache-mxnet) post for more details on embedding, followed by a RNN (GRU / LSTM). The RNN unit returns an output as well as hidden state. The output produced by the RNN is passed to a decoder (dense unit) which predicts the next character in the neural network. We also have a begin state function that initialise the initial hidden state of the model.

### Training the neural network

After defining the network we can train the neural network as follows

```python 
def trainGluonRNN(epochs,train_data,seq=seq_length):
    best_val = float("Inf")
    for epoch in range(epochs):
        total_L = 0.0
        start_time = time.time()
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx = context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, seq_length)):
            data, target = get_batch(train_data, i,seq)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target)
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and bptt size to balance it.
            gluon.utils.clip_global_norm(grads, clip * seq_length * batch_size)

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()
        model.save_params(rnn_save)
```

A each epoch's begining, we initialise the hidden units to zero state. During training of each batch we detach hidden unit from computational graph as we dont want to back propogate gradient beyond the sequence lenght. We also scale the gradient by multiplying with sequence lenght and batch size

### Text generation.

After training for 200 epochs, We can generate random text. The following python code generates random text from the learnt model

```python
# a nietzsche like text generator
import sys
def generate_random_text(model,input_string,seq_length,batch_size,sentence_length):
    count = 0
    new_string = ''
    cp_input_string = input_string
    while count < sentence_length:
        idx = [char_indices[c] for c in input_string]
        if(len(input_string) != seq_length):
            print(len(input_string))
            raise ValueError('there was a error in the input ')
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx=context)
        sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T
                                ,ctx=context)
        output,hidden = model(sample_input,hidden)
        index = mx.nd.argmax(output, axis=1)
        index = index.asnumpy()
        count = count + 1
        new_string = new_string + indices_char[index[-1]]
        input_string = input_string[1:] + indices_char[index[-1]]
    print(cp_input_string + new_string)

```


## Generative adversial network (GAN)

Generative adversial network is neural network model based on game theory zero-sum game. It typically consists of two different neural network namely Discriminator and Generator, where each network tries to outwin the other. Let us consider a inutitive example to understand GAN network. 
![Alt text](images/GAN_SAMPLE.png?raw=true "Generative Adversial Network")

Suppose there is bank (discriminator) that decides whether a give currency is real or fake. A counterfeit team which wants to produce fake currency can learn all the details of currency by looking at few real curreny notes. Then they can produce a similar fake currency note and give it to the bank. The bank runs its own secret algorithm to determine whether the give currency note is real or fake. For a given fake currency note given by counterfeit, the bank can reject it by saying it as fake and gived a partial reasoning about it. The counterfeit team can take the 'reason / loss' and can try to improve their model. After mutliple iterations, the bank cannot determine the difference between real and fake currency. This is basic idea behind GAN . Let start implementing a simple GAN network.

I encourage you to download [the notebook](https://github.com/sookinoby/generative-models/blob/master/GAN.ipynb) where we've created and run all this code, and play with it! Adjust the hyperparameters and experiment with different approaches to neural network architecture.

### Preparing the DataSet

We use a library called [brine](https://docs.brine.io/getting_started.html) to download our dataset. Brine has various dataset and we can choose the dataset to download. So to install and download dataset do the following

1. pip install brine-io
2. brine install jayleicn/anime-faces

Anime-faces contains over 100,000 anime images collected from internet 

Once the dataset is downloaded , you can load the dataset as follow

```python
# brine for loading anime-faces dataset
import brine
anime_train = brine.load_dataset('jayleicn/anime-faces')
```

We also need to normalise the pixel value of each image to [-1 to 1] and also channel the ordering of image from  (width X height X channels) to (channels X width X height ) since MxNet expects this format.

```python
def transform(data, target_wd, target_ht):
    # resize to target_wd * target_ht
    data = mx.image.imresize(data, target_wd, target_ht)
    # transpose from (target_wd, target_ht, 3) 
    # to (3, target_wd, target_ht)
    data = nd.transpose(data, (2,0,1))
    # normalize to [-1, 1]
    data = data.astype(np.float32)/127.5 - 1
    return data.reshape((1,) + data.shape)

# Read images, call the transform function, attach it to list
def getImageList(base_path,training_folder):
    img_list = []
    for train in training_folder:
        fname = base_path + train.image
        img_arr = mx.image.imread(fname)
        img_arr = transform(img_arr, target_wd, target_ht)
        img_list.append(img_arr)
    return img_list

base_path = 'brine_datasets/jayleicn/anime-faces/images/'
img_list = getImageList('brine_datasets/jayleicn/anime-faces/images/',training_fold)
```


### Designing the network

We need to design two seperate networks i.e. discriminator network and a generator network. Generator takes in a shape of (batchsize X N ) dimension random vector and converts it image of shape (batchsize X channels X width X height). It use trasponse convolutions to upscale the one dimension input vector into a image. This is very similar to a decoder unit in an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) trying to map a lower dimension vector into higher dimensional vector representation. Below is the snippet of generator network

```python  
with netG.name_scope():
    # input is Z, going into a convolution
    netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 4 x 4
    netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 8 x 8
    netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 16 x 16
    netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # state size. (ngf*8) x 32 x 32
    netG.add(nn.Conv2DTranspose(nc, 4, 2, 1))
    netG.add(nn.Activation('tanh')) # use tanh , we need an output that is between -1 to 1, not 0 to 1 
    # Rememeber the input image is normalised between -1 to 1, so should be the output
    # state size. (nc) x 64 x 64
```

Discriminator are basically a binary image classification network that maps the image of shape (batchsize X channels X width x height) into a lower dimension vector of shape (batchsize X 1). This is similar to a encoder that converts a higher dimension image representation into lower one. Below is the snippet of generator network

```python
with netD.name_scope():
    # input is (nc) x 64 x 64
    netD.add(nn.Conv2D(ndf, 4, 2, 1))
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 32 x 32
    netD.add(nn.Conv2D(ndf * 2, 4, 2, 1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 16 x 16
    netD.add(nn.Conv2D(ndf * 4, 4, 2, 1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 8 x 8
    netD.add(nn.Conv2D(ndf * 8, 4, 2, 1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # state size. (ndf) x 4 x 4
    netD.add(nn.Conv2D(1, 4, 1, 0))
```
### Training the GAN network

The training of GAN network is not straight forward but it is simple enough. Below the illustration of training process.  ![Alt text](images/GAN_Model.png?raw=true "GAN training").  The real images are given a label one and the fake images are given a label zero

```python
#real label is the labels of real image
real_label = nd.ones((batch_size,), ctx=ctx)
#fake labels is label associated with fake image
fake_label = nd.zeros((batch_size,),ctx=ctx)
```
#### Training the discriminator

 A real image is also passed to discriminator to determine if it is real or fake and loss errD_real.

 ```python
# train with real image
output = netD(data).reshape((-1, 1))
#The loss is a real valued number
errD_real = loss(output, real_label)
``` 

The random noise z is passed to generator network to generate a random image.This image is then passed to the discriminator to classify if it as real (1) or fake(0) producing a loss, errD_fake.
 
 ```python            
#train with fake image, see the what the discriminator predicts
#creates fake imge
fake = netG(latent_z)
# pass it to discriminator
output = netD(fake.detach()).reshape((-1, 1))
errD_fake = loss(output, fake_label)
 ```

The total error is backprogapagated to tune the weights of discrimnator.

 ```python
#compute the total error for fake image and the real image
errD = errD_real + errD_fake
#improve the discriminator skill by back propagating the error
errD.backward()
```

#### Training the generator

The same random noise to generate a fake image. Then we pass the fake image to the discriminator network to obtain the classification output, and loss is calculated. The loss is then used fine tune the network

```python
fake = netG(latent_z)
output = netD(fake).reshape((-1, 1))
errG = loss(output, real_label)
errG.backward()
```

### Generating new fake images

We can use the generator network to create new fake images by provding 100 dimension random input to the network.

 ![Alt text](images/GAN_image.png?raw=true "GAN generated images").
```
#Lets generate some random images
num_image = 8
for i in range(num_image):
    # randome input for the generating images
    latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size, 1, 1), ctx=ctx)
    img = netG(latent_z)
    plt.subplot(2,4,i+1)
    visualize(img[0])
plt.show()
```


## Conculsion
Generative models opens new opportunities for deep learning.  We explored some the popular generative models for text as well as image. We learnt basics of RNN and how RNN can be constructed using feed forward neural network. We also used RNN to generate text similar to Friedrich Nietzsche using LSTM.
Then we learnt of GAN models and generated images similar to that of input data (Anime Characters). 