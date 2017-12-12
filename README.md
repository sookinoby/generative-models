# Generative Models using Apache MXNet
In our previous notebooks, we used a deep learning technique called Convolution Neural Network (CNN) to classify text and images.  A CNN is generally referred as a Discriminative Model.  A Discriminative Model tries to create a decision boundary to classify a given input signal (data).  
Deep learning models, in recent times have been used to generate data based on the given input signal – these are called Generative Models.  A Generative Model tries to understand the underlying distribution, it can also generate new data or classify a given input data.  We have explained the difference between Generative and Discriminative model in the “Generative Models” section below. 
Generative models are powerful and provide insights about the underlying phenomenon that generates the data, therefore, they can generate data similar to the input data. Generative Models can be used to: 
1. Predict  the probability of a word or character given the previous word or character.
2. Produce a new song or combine two genre of songs to create an entirely different song, and  synthesize new images from existing images are some examples of generative models. 
3. Up [sample images](https://arxiv.org/pdf/1703.04244.pdf) and much more.
In general, Generative Models can be used on any form of data to learn the underlying distribution and produce new data / augment existing data.
In this tutorial, we are going to build Generative Models, using Apache MXNet gluon API, that will predict the next character/word. In other words, we are going to build our own version of [swiftkey](https://blog.swiftkey.com/swiftkey-debuts-worlds-first-smartphone-keyboard-powered-by-neural-networks/). 
Then we will also talk about the following topics: 
The building blocks of a Recurrent Neural Network (RNN)
Implementing an unrolled version of RNN, LSTM(Long short term memory), GAN (generative adversarial neural network) using MXNet Gluon API. 

You need to have a basic understanding of Recurrent Neural Network(RNN), Activation Units, Gradient Descent, and NumPy to understand this tutorial. 
By the end of the notebook, you will be able to:  
1. Understand Generative Models
2. Know the limitations of a Feed Forward Neural Network
3. Understand the idea behind RNN and LSTM.
4. Install MXNet with Gluon API.
5. Prepare datasets to train the Neural Network.
6. Implement a basic RNN using Feed Forward Neural Network
6. Implement a Generative Model to auto generate text using Gluon API
7. Implement a Generative Adaptive Neural Network

## Generative models- Discriminative models: Martians vs Humans
Let us take a simple example to understand our dataset better. Let’s say that we have to classify Martians and Humans based on their heights (in centimeters). Below is the sample data set

Martian - 250,260,270,300,220,260,280,290,300,310 <br />
Human - 160,170,180,190,175,140,180,210,140,200 <br />

If we train a Discriminative Model, it will only learn a decision boundary (at height 200 cm). ![Alt text](images/martians-chart5_preview.jpeg?raw=true "Unrolled RNN") <br />
The model doesn’t care about the underlying distribution that generates the data.

On the other hand, a Generative Model will learn the underlying distribution for Martian (mean =274, std= 8.71) and Human (mean=174, std=7.32).  ![Alt text](images/humans_mars.png?raw=true "Unrolled RNN")<br />

By extending this model, we can generate new Martians and Humans, or a new interbreed species (humars). We can also use this model for classifying Martians and Humans.

## Limitations of Feed Forward Neural Network

Although Feed Forward Neural Network (including Convolution Neural Network)have shown great accuracy in classifying sentences and text, it cannot store long term dependencies in its memory(hidden state).  For example, whenever we think about KFC chicken, our brain immediately interprets it as ‘hot’ and ‘crispy’. This is because our brains can remember the context of a conversation, and retrieve those contexts whenever it needs. A Feed Forward Neural Network doesn't have the capacity to interpret the context. 
Any RNN cell will produce two outputs, one is the actual output and the other is the hidden state. If RNN is a person talking on the phone, the actual output is the words spoken and the hidden state is the context in which the person utters the word.  ![Alt text](images/sequene_to_sequence.png?raw=true "Sequence to Sequence model") <br />
The yellow arrows are the hidden state and the red arrows are the output.
 Convolutional Neural Network can only remember spatial information for a small local neighbour(size of convolution kernels) and cannot model sequential data (data with definitive ordering, like structure of a language).  Let's see a simple example to understand long term dependencies.

```python
"<html>
<head>
<title>
RNN, Here I come.
 </title>
 </head> <body>Html is amazing, but I should not forget the end tag</body>
 </html>"
 ```
Let’s say we are building a predictive text editor, which helps users auto-complete the current word by using the words in the current document and/or your previous typing habit.  The model should remember long term dependencies like start tag ‘<html>’ and end tag ’</html>’. A Convolutional Neural Network does not have provision to remember long term context/information. A RNN can remember the context using its internal ‘memory’. If RNN is a person, this is how they think “Hey I saw ‘<html>’ tag, then <title> tag, I might need to close the ‘<title>’ tag before closing the ‘<html>’ tag.”
## Intuition behind RNN.

Let’s say we have to predict the 4th character given the first 2 characters, to do that we can design a simple neural network as shown below ![Alt text](images/unRolled_rnn.png?raw=true "Unrolled RNN") <br />
 This is basically a Feed Forward Network where the weights WI(green arrow), WH(yellow arrow) are shared between some of the layers. This is an unrolled version of RNN  and this type of RNN are generally referred as many-to-one RNN, since N inputs (3 characters) are used to predict one character. This can be designed using MxNet as follows:

```python
class UnRolledRNN_Model(Block):
  # This is the initialisation of UnRolledRNN
    def __init__(self,vocab_size, num_embed, num_hidden,**kwargs):
        super(UnRolledRNN_Model, self).__init__(**kwargs)
        self.num_embed = num_embed
        self.vocab_size = vocab_size
        
        # Use name_scope to give child Blocks appropriate names.
        # It also allows sharing parameters between blocks recursively.
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
Basically this neural network has 3 embedding layers (emb) for each character, followed by 3 dense layers. They are: dense 1 (shared weight) for inputs,  dense 2 (dense layer) and 1 dense layer (dense 3) that produces the output. We also do some MXNET array addition to combine inputs.

Similar to many-to-one RNN, there are other types of RNN models, including the popular sequence-to-sequence RNN: 
![Alt text](images/loss.png?raw=true"Sequence to Sequence model") <br />

Here N inputs (3 characters) are mapped onto 3 outputs, this helps the model to train faster as we “loss” (difference in the prediction and the actual output) at each time instant. Instead of one loss at the end,  we can see loss1, loss2, …. , so each loss can be used to fine tune the network. 
We use [Binary Cross Entropy Loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss) function in our model.

So that we get a better feedback (backpropagation) when training our model. 
This model can be folded back and succinctly represented like this: 
 ![Alt text](images/RNN.png?raw=true "RNN") <br />

The above representation also makes the math behind the model easy to understand: 

```python
hidden_state_at_t = (WI x input + WH x previous_hidden_state)
```

The are some limitations with basic RNN. For example, let us take a document that has 1000 words " I was born in france during world war.... So I can speak french". A simple RNN may not be able to understand the context between "being born in france" and "I can speak french" because they can be far apart (temporally distant) in a given document.
RNN doesn’t provide the capability to forget irrelevant context in between the phrases. RNN gives more importance to the previous hidden state because it cannot give preference to the arbitrary (t-k) hidden state, where t is the current time step and k is the number greater than 0.  Training an RNN on a long sequence of words can cause gradient in backpropagation to vanish (when gradient is less than one) or to explode (when gradient is larger than 1), as [back propagation[(http://neuralnetworksanddeeplearning.com/chap2.html) basically multiplies the gradients along the computational graph in reverse direction. A detailed explanation of problems with RNN is given [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.421.8930&rep=rep1&type=pdf).  
## Long short term memory (LSTM)
In order to address the problems with basic RNN German researchers, Sepp Hochreiter and Juergen Schmidhuber proposed [Long short term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (LSTM, a complex RNN unit) as a solution to the vanishing / exploding gradient problem.  A beautifully illustrated simpler version of LSTM can be found [here](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714). We can see that each unit of LSTM has a small neural network that decides the amount of information it needs to remember (memory) from the previous time step. 
The diagram below illustrates the LSTM model: 
 ![Alt text](images/lstm.png?raw=true "RNN") <br />

# Preparing your environment

If you're working in the AWS Cloud, you can save yourself the installation management by using an [Amazon Machine Image](https://aws.amazon.com/marketplace/pp/B01M0AXXQB#support), pre-configured for deep learning.  If you have done this, then skip steps 1-5 below. 

Note that if you are using a Conda environment, remember to install pip inside conda by typing 'conda install pip' after you activate an environment.  This will save you a lot of problems down the road.

Here's how to get set up: 

1. Install [Anaconda](https://www.continuum.io/downloads), a package manager. It is easier to install Python libraries using Anaconda.
2. Install [scikit learn](http://scikit-learn.org/stable/install.html), a general-purpose scientific computing library. We'll use this to pre-process our data. You can instal it with 'conda install scikit-learn'.
3. Then grab the Jupyter Notebook, with 'conda install jupyter notebook'.
4. And then, get [MXNet](https://github.com/apache/incubator-mxnet/releases), an open source deep learning library. The python notebook was tested on  0.12.0 version of MxNet, and  you can install using pip as follows: pip install mxnet==0.12.0

Once you active the anaconda environment, type these commands in it: 
1. conda install pip 
2. pip install opencv-python
3. conda install scikit-learn
4. conda install jupyter notebook
5. pip install mxnet==0.12.0

# Generative Models for characters
Here’s the notebook for this part of the tutorial.
 You can download the MXNet notebook for this part of the tutorial [here](https://github.com/sookinoby/generative-models/blob/master/Test-rnn.ipynb), where we've created and run all this code, and play with it!
Adjust the hyperparameters and experiment with different approaches to neural network architecture.

## Preparing the DataSet

We will use text literary work of [Friedrich Nietzsche](https://en.wikipedia.org/wiki/Friedrich_Nietzsche) as our dataset. 
You can download the data set [here](https://s3.amazonaws.com/text-datasets/nietzsche.txt). You are free to use any other dataset including your own chat history or you can also download some datasets from this [site](https://cs.stanford.edu/people/karpathy/char-rnn/).
The data set nietzsche.txt consists of 600901 characters out of which 86 are unique. We need to convert the entire text to a sequence of numbers. 

```python
chars = sorted(list(set(text)))
#maps character to unique index e.g. {a:1,b:2....}
char_indices = dict((c, i) for i, c in enumerate(chars))
#maps indices to character (1:a,2:b ....)
indices_char = dict((i, c) for i, c in enumerate(chars))
#convert the entire text into sequence
idx = [char_indices[c] for c in text]
```

### Preparing dataset for Un-rolled RNN
 We need to convert the dataset in such a way that the input will be the first three characters, and the output will be the 4th character. Let’s say we have a sentence ‘I_love_mxnet’, this will be converted into the following input ![Alt text](images/unroll_input.png?raw=true "unrolled input") <br />


 ```python
 #Input for neural network(our basic rnn has 3 inputs, n samples)
cs=3
c1_dat = [idx[i] for i in range(0, len(idx)-1-cs, cs)]
c2_dat = [idx[i+1] for i in range(0, len(idx)-1-cs, cs)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-1-cs, cs)]
#The output of rnn network (single vector)
c4_dat = [idx[i+3] for i in range(0, len(idx)-1-cs, cs)]
#Stacking the inputs to form (3 input features )
x1 = np.stack(c1_dat[:-2])
x2 = np.stack(c2_dat[:-2])
x3 = np.stack(c3_dat[:-2])

# Concatenate to form the input training set
col_concat = np.array([x1,x2,x3])
t_col_concat = col_concat.T

```
We also batchify the training set in batches of 32, so each training instance is of shape 32 X 3. Batchifying the input helps us to train faster.

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

### Preparing the dataset for gluon RNN

This is very similar to preparing the dataset for Un-rolled RNN, expect for shape of input. The dataset should be ordered in the shape (number of example X batch_size). For example, let us consider the sample dataset below:
![Alt text](images/batch_reshape.png?raw=true "batch reshape") <br />
In the above image, the input sequence is converted to batch of size 3, and then into 2 separate input sequence of length 2. By transforming it this way, it is very easy to generate arbitrary length input sequence, say 5. During our training, we use a input sequence length of 15. This is a hyperparameter and may require fine tuning for best output.

## Designing RNN in Gluon
Next, we define a class which allows us to create two types of RNN namely GRU (Gated Recurrent Unit) and LSTM. GRU is a simpler version of LSTM, and also performs as good as LSTM. You can find a comparison study [here](https://arxiv.org/abs/1412.3555). Below is the Python snippet:

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
The constructor of class creates few neural units that will be used in our forward pass. The forward pass is the method that will be called during our training to generate the  loss associated with the training data.
The forward pass function in the GluonRNNModel creates an [embedding layer](https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.nn.Embedding) for the input character. You can look at our[previous blog post](https://www.oreilly.com/ideas/sentiment-analysis-with-apache-mxnet) for more details on embedding. The output of the embedding layer is  provided as aninput to the RNN ([GRU](https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.rnn.GRU) / [LSTM](https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.rnn.LSTM) ) layer. The RNN unit returns an output as well as hidden state. The output produced by the RNN is passed to a decoder (dense unit) which predicts the next character in the neural network and also generate the loss. We also have a “begin state” function that initializes the initial hidden state of the model.

## Training the neural network

After defining the network. we have to train the neural network for it to learn the underlying distribution.

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

At the beginning of each epoch, we initialise the hidden units to zero state. While training each batch, we we detach the hidden unit from computational graph so that we don’t back propagate the gradient beyond the sequence length (15 in our case). If we don’t detach the hidden state, the gradient is passed to the beginning of hidden state (t=0).  We also scale the gradient by multiplying with sequence length and batch size to normalise it. L.backward backpropagates the loss to fine tune the weights. 

## Text generation.

After training for 200 epochs, we can generate random text. The following python code generates random text. Here we initialize the hidden state and pass a initial input string. Then we recursively pass the generated output back into the model to make prediction.  

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
Next we will look into generative models for images and specially GAN (Generative Adversarial network)
# Generative Adversarial network (GAN)

[Generative adversarial network](https://arxiv.org/abs/1406.2661) is a neural network model based on game theory [zero-sum game](https://en.wikipedia.org/wiki/Zero-sum_game). It typically consists of two different neural networks called Discriminator and Generator, where each network tries to outperform the other. Let us consider an example to understand GAN network. 
![Alt text](images/GAN_SAMPLE.png?raw=true "Generative Adversarial Network")

Let’s assume that there is a bank (discriminator) that detects whether a given currency is real or fake using machine learning. If a fraudster builds a machine learning model to counterfeit fake currency notes by looking at the real currency notes, and deposits them in bank. The bank will identify the currencies deposited as fake. 
If the bank tells the fraudster why they classified these notes as “fake” then he can improve his model based on those reasons. After multiple iterations, the bank cannot find the difference between the “real” and “fake” currency. This is the basic idea behind GAN . 
Let start implementing a simple GAN network.

I encourage you to download [the notebook](https://github.com/sookinoby/generative-models/blob/master/GAN.ipynb).
You are welcome to adjust the hyperparameters and experiment with different approaches to neural network architecture.

## Preparing the DataSet

We use a library called [brine](https://docs.brine.io/getting_started.html) to download our dataset. Brine has many datasets, so we can choose the dataset that we want to download. To install and download dataset do the following:

1. pip install brine-io
2. brine install jayleicn/anime-faces

I am using Anime-faces dataset for this tutorial. The Anime-faces contains over 100,000 anime images collected from internet.  

Once the dataset is downloaded , you can load the dataset using the following code: 

```python
# brine for loading anime-faces dataset
import brine
anime_train = brine.load_dataset('jayleicn/anime-faces')
```

We also need to normalise the pixel value of each image to [-1 to 1] and also channel the ordering of image from (width X height X channels) to (channels X width X height ). I am doing this because MxNet expects this format.

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


## Designing the network
We need to design two separate networks i.e. discriminator network and a generator network. Generator takes a random vector of shape (batchsize X N ), where N is an integer,  as input and converts it to a image of shape (batchsize X channels X width X height). It uses [transpose convolutions](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#no-zero-padding-unit-strides-transposed) to upscale the input vectors. This is very similar to a decoder unit in an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) trying to map a lower dimension vector into higher dimensional vector representation. Below is the snippet of a generator network

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
Discriminator is a binary image classification network that maps the image of shape (batchsize X channels X width x height) into a lower dimension vector of shape (batchsize X 1). This is similar to an encoder that converts a higher dimension image representation into a lower one. Below is the snippet of generator network:

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
## Training the GAN network
The training of a GAN network is not straightforward but it is simple. The below diagram illustrates the training process.  ![Alt text](images/GAN_Model.png?raw=true "GAN training") <br />  The real images are given a label one and the fake images are given a label zero

```python
#real label is the labels of real image
real_label = nd.ones((batch_size,), ctx=ctx)
#fake labels is label associated with fake image
fake_label = nd.zeros((batch_size,),ctx=ctx)
```
### Training the discriminator

 A real image is also passed to the discriminator, to determine if it is real or fake and the loss associated with the prediction is calculated as errD_real.

 ```python
# train with real image
output = netD(data).reshape((-1, 1))
#The loss is a real valued number
errD_real = loss(output, real_label)
``` 

In the next step, a random noise z is passed to the generator network to produce a random image. This image is then passed to the discriminator to classify it as real (1) or fake(0), thereby  producing a loss, errD_fake.
 
 ```python            
#train with fake image, see what the discriminator predicts
#creates fake image
fake = netG(latent_z)
# pass it to the discriminator
output = netD(fake.detach()).reshape((-1, 1))
errD_fake = loss(output, fake_label)
 ```

The total error is backpropagated to tune the weights of the discriminator.

 ```python
#compute the total error for fake image and the real image
errD = errD_real + errD_fake
#improve the discriminator skill by back propagating the error
errD.backward()
```

### Training the generator

The random noise vector used in the training of discriminator is used again to generate a fake image. Then we pass the fake image to the discriminator network to obtain the classification output, and loss is calculated. The loss is then used to fine tune the network.

```python
fake = netG(latent_z)
output = netD(fake).reshape((-1, 1))
errG = loss(output, real_label)
errG.backward()
```

## Generating new fake images

We can use the generator network to create new fake images by providing 100 dimension random input to the network.

 ![Alt text](images/GAN_image.png?raw=true "GAN generated images")<br />
```
#Let’s generate some random images
num_image = 8
for i in range(num_image):
    # random input for generating images
    latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size, 1, 1), ctx=ctx)
    img = netG(latent_z)
    plt.subplot(2,4,i+1)
    visualize(img[0])
plt.show()
```


# Conclusion
Generative models opens new opportunities for deep learning.  We explored some of the popular generative models for text and image. We learnt basics of RNN and how RNN can be constructed using feed forward neural network. We also used RNN to generate text similar to Friedrich Nietzsche using LSTM.
Then we learnt about GAN models and generated images similar to input data (Anime Characters).
