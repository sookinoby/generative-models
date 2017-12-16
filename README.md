# Generative Models using Apache MXNet
In our previous notebooks, we used a deep learning technique called Convolution Neural Network (CNN) to classify text and images.  A CNN is an example of a Discriminative Model, which tries to create a decision boundary to classify a given input signal (data).

Deep learning models in recent times have been used to create even more powerful and useful tools called Generative Models. A Generative Model goes beyond a simple decision boundary to understand the underlying distributions of values. From this insight, a generative model can also generate new data or classify a given input data. Some examples of uses for generative models include:
1. Predicting the probability of a word or character given the previous word or character. This is a widespread practice in mobile texting that suggest completions when the user is typing. One exemplary system is [SwiftKey](https://blog.swiftkey.com/swiftkey-debuts-worlds-first-smartphone-keyboard-powered-by-neural-networks/).
2. Producing a new song or combine two genre of songs to create an entirely different song, and  synthesize new images from existing images are some examples of generative models.
3. [Upgrading images](https://arxiv.org/pdf/1703.04244.pdf) to a higher resolution, resolving fuzziness.
And much more. In general, Generative Models can be used on any form of data to learn the underlying distribution, produce new data, and augment existing data.

In this tutorial, we are going to build Generative Models, using Apache MXNet gluon API, for the first application just listed: predicting the next character in an incoming stream.

We will also talk about the following topics:
* The difference between generative and discriminative models
* The building blocks of a Recurrent Neural Network (RNN)
* Implementing an unrolled version of RNN to understand its relationship with feed forward neural network, then Long short-term memory (LSTM) and Gated Recurrent Uniy(GRU) RNN and followed by Generative Adversarial Neural Network (GAN) using the MXNet Gluon API.


You need to have a basic understanding of Recurrent Neural Network(RNN), Activation Units, Gradient Descent, Back Propagation and NumPy to understand this tutorial.
By the end of the notebook, you will be able to:
1. Understand Generative Models
2. Know the limitations of a Feed Forward Neural Network
3. Understand the idea behind RNN and LSTM
4. Install MXNet with Gluon API
5. Prepare datasets to train the Neural Network
6. Implement a basic RNN using Feed Forward Neural Network
6. Implement a Generative Model to auto generate text using Gluon API
7. Implement a Generative Adversarial Neural Network (GAN) Neural Network

First, we will discuss on the idea behind Generative models, followed by limitations of feed forward neural network. Next, we will implementa basic RNN using feed forward neural network provide good insight into working of RNN. Then we design a power RNN with LSTM and GRU layers using MxNet gluon API. Next, we implement GAN which can generate new image from exsiting images. By the end of tutorial, you will be able to implement other cool generative models using Gluon API.

### How Generative Models Go Further Than Discriminative Models
We can grasp the power of Generative Models through a trivial example. The heights of human beings follow a normal distribution, showing up as a bell-shaped curve on a graph. Martians tend to be much taller than humans (trust me on this) but also follow a normal distribution. So let's measure some humans and Martians and feed their heights into a discriminative model, followed by a generative model. Our sample data set is;

Martian - 250,260,270,300,220,260,280,290,300,310 <br />
Human - 160,170,180,190,175,140,180,210,140,200 <br />

If we train a Discriminative Model, it will only learn a decision boundary. Let's suppose it recognizes that Martians are taller than 200 cm while Humans are shorter. This actually misclassifies one human, but the accuracy is quite good overall. So the discriminative model is useful for classifying new beings of one planet or another that come along, but not for the more powerful applications listed at the beginning of this article. In particular, the model doesn’t care about the underlying distribution of data. ![Alt text](images/martians-chart5_preview.jpeg?raw=true "Unrolled RNN") <br />

In contrast, a generative model will learn the underlying distribution for Martian (mean =274, std= 8.71) and Human (mean=174, std=7.32).  ![Alt text](images/humans_mars.png?raw=true "Unrolled RNN")<br />

By extending this model, we can generate new Martians and Humans, or a new interbreed species (humars). We can also use this model for classifying Martians and Humans, just like the discriminative model.

### The Need For Hidden State

Although Feed Forward Neural Networks, including Convolution Neural Networks, have shown great accuracy in classifying sentences and text, they cannot store long term dependencies in memory (hidden state). For example, whenever an average American thinks about KFC chicken, her brain immediately thinks of it as "hot" and "crispy". This is because our brains can remember the context of a conversation, and retrieve those contexts whenever it needs. A Feed Forward Neural Network doesn't have the capacity to interpret the context. In a CNN can learn temporal context, local group of a neighbors within the size of its convolution kernels. So it cannot model sequential data (data with definitive ordering, like structure of a language). A abstract view of feed forward neural network is show below <br /> ![Alt text](images/ffn_rnn.png?raw=true "Sequence to Sequence model")


An RNN is more versatile. It's cells accept weighted input and produce both weighted output (WO) and weighted hidden state (WH). If an RNN represents a person talking on the phone, the weighted output is the words spoken and the weighted hidden state is the context in which the person utters the word.  ![Alt text](images/sequene_to_sequence.png?raw=true "Sequence to Sequence model") <br />

The yellow arrows are the hidden state and the red arrows are the output.

A simple example can help us understand long term dependencies.

```python
<html>
<head>
<title>
RNN, Here I come.
 </title>
 </head> <body>HTML is amazing, but I should not forget the end tag.</body>
 </html>
 ```
Let’s say we are building a predictive text editor, which helps users auto-complete the current word by using the words in the current document and perhaps the users' prior typing habits.  The model should remember long term dependencies like the need for the start tag <html> and to matched by an end tag </html>. A CNN does not have provision to remember long term context like that. A RNN can remember the context using its internal "memory," just as a person might think “Hey, I saw an <html> tag, then a <title> tag, so I need to close the <title> tag before closing the <html> tag.”

### The intuition behind RNNs

Suppose we have to predict the 4th character in a stream of text, given the first 3 characters. To do that, we can design a simple feed forward neural network as in the following figure. ![Alt text](images/unRolled_rnn.png?raw=true "Unrolled RNN") <br />

This is basically a Feed Forward Network where the weights WI (green arrows) and WH (yellow arrows) are shared between some of the layers. This is an unrolled version of Vanilla RNN, generally referred to as a many-to-one RNN because multiple inputs (3 characters, in this case) are used to predict one character. The RNN can be designed using MxNet as follows:

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
        addition_hidden = self.dense2(addition_result) # yellow arrow in diagram (WH)
        addition_result_2 = F.add(addition_hidden,c3_hidden) # Total c1 + c2 + c3
        final_output = self.dense3(addition_result_2)   # The red arrow in diagram (WO)
        return final_output
  ```
Basically, this neural network has 3 embedding layers (emb) for each character, followed by 3 dense layers, named dense1 (shared weight) taking the inputs,  dense2 (an intermediate layer), and dense3, which produces the output. We also do some MXNet array addition to combine inputs.

In addition to the many-to-one RNN, there are other types of RNN that process such memory-based applications, including the popular sequence-to-sequence RNN:
![Alt text](images/loss.png?raw=true"Sequence to Sequence model") <br />


Here N inputs (3 characters) are mapped onto N outputs. This helps the model train faster, because we measure loss (the difference between the predicted value and the actual output) at each time instant. Instead of one loss at the end, we can see loss1, loss2, So that we get a better feedback (backpropagation) when training our model.
We use [Binary Cross Entropy Loss](https://mxnet.incubator.apache.org/api/python/gluon/loss.html#mxnet.gluon.loss.SigmoidBinaryCrossEntropyLoss) in our model.

This model can be folded back and succinctly represented like this:  <br />
 ![Alt text](images/RNN.png?raw=true "RNN")  <br />

The above representation also makes the math behind the model easy to understand:

```python
hidden_state_at_t = (WI x input + WH x previous_hidden_state)
```

There are some limitations with vanilla RNN. For example, let us take a document that has 1000 words including the sentences "I was born in France during the world war" and "So I can speak French." A vanilla RNN may not be able to understand the context between being "born in France" and "I can speak French" because they can be far apart (temporally distant) in a given document.

RNN doesn’t provide the capability (at least in practice) to forget irrelevant context in between the phrases. RNN gives more importance to the most previous hidden state because it cannot give preference to the arbitrary (t-k) hidden state, where t is the current time step and k is the number greater than 0. This is because training an RNN on a long sequence of words can cause the gradient to vanish (when gradient is small) or to explode (when gradient is large) during backpropagation, because [backpropagation](http://neuralnetworksanddeeplearning.com/chap2.html) basically multiplies the gradients along the computational graph in reverse direction. A detailed explanation of problems with RNN is given [here](http://citeseerx.ist.psu.edu/viewdoc/download?doi=10.1.1.421.8930&rep=rep1&type=pdf).

### Long short-term memory (LSTM)
In order to address the problems with vanilla RNN, the two German researchers Sepp Hochreiter and Juergen Schmidhuber proposed [Long short-term memory](http://www.bioinf.jku.at/publications/older/2604.pdf) (LSTM, a complex RNN unit) as a solution to the vanishing/exploding gradient problem.  A beautifully illustrated simpler version of LSTM can be found [here](https://medium.com/mlreview/understanding-lstm-and-its-diagrams-37e2f46f1714). In abstract sense, We can think LSTM unit as a small neural network that decides the amount of information it needs to preserve (memory) from the previous time step.

## Implementing an LSTM

Now we can try creating our own simple character predictor.

### Preparing your environment

If you're working in the AWS Cloud, you can save yourself a lot of installation work by using an [Amazon Machine Image](https://aws.amazon.com/marketplace/pp/B01M0AXXQB#support), pre-configured for deep learning.  If you have done this, skip steps 1-5 below.
(AO: That's skipping everything. I assumed the fifth step was the series of shell commands.)

If you are using a Conda environment, remember to install pip inside conda by typing 'conda install pip' after you activate an environment.  This will save you a lot of problems down the road.

Here's how to get set up:

1. Install [Anaconda](https://www.continuum.io/downloads), a package manager. It is easier to install Python libraries using Anaconda.
2. Install [scikit-learn](http://scikit-learn.org/stable/install.html), a general-purpose scientific computing library. We'll use this to pre-process our data. You can install it with 'conda install scikit-learn'.
3. Grab the Jupyter Notebook, with 'conda install jupyter notebook'.
4. Get [MXNet](https://github.com/apache/incubator-mxnet/releases), an open source deep learning library. The Python notebook was tested on version 0.12.0 of MxNet, and  you can install using pip as follows: pip install mxnet==0.12.0
5. After you activate the anaconda environment, type these commands in it:

```bash
conda install pip
pip install opencv-python
conda install scikit-learn
conda install jupyter notebook
pip install mxnet==0.12.0
```

6. You can download the MXNet notebook for this part of the tutorial [here](https://github.com/sookinoby/generative-models/blob/master/Test-rnn.ipynb), where we've created and run all this code, and play with it! Adjust the hyperparameters and experiment with different approaches to neural network architecture.

### Preparing the Data Set

We will use a work of [Friedrich Nietzsche](https://en.wikipedia.org/wiki/Friedrich_Nietzsche) as our dataset.
You can download the data set [here](https://s3.amazonaws.com/text-datasets/nietzsche.txt). You are free to use any other dataset, such as your own chat history, or you can download some datasets from this [site](https://cs.stanford.edu/people/karpathy/char-rnn/).
The data set nietzsche.txt consists of 600901 characters, out of which 86 are unique. We need to convert the entire text to a sequence of numbers.

```python
chars = sorted(list(set(text)))
#maps character to unique index e.g. {a:1,b:2....}
char_indices = dict((c, i) for i, c in enumerate(chars))
#maps indices to characters (1:a,2:b ....)
indices_char = dict((i, c) for i, c in enumerate(chars))
#convert the entire text into sequence
idx = [char_indices[c] for c in text]
```

### Preparing dataset for Unrolled RNN
Our goal is to convert the data set to a series of inputs and output. Each sequence of three characters from the input stream will be stored as the three input characters to our model, with the next character being the output we are trying to train our model to predict. For instance, we would translate the string "I_love_mxnet" into the following set of inputs and outputs. ![Alt text](images/unroll_input.png?raw=true "unrolled input") <br />
(AO: The table is missing the final row.)

The code to do the conversion follows.

 ```python
 #Input for neural network(our basic rnn has 3 inputs, n samples)
cs=3
c1_dat = [idx[i] for i in range(0, len(idx)-1-cs, cs)]
c2_dat = [idx[i+1] for i in range(0, len(idx)-1-cs, cs)]
c3_dat = [idx[i+2] for i in range(0, len(idx)-1-cs, cs)]
#The output of rnn network (single vector)
c4_dat = [idx[i+3] for i in range(0, len(idx)-1-cs, cs)]
#Stacking the inputs to form3 input features
x1 = np.stack(c1_dat[:-2])
x2 = np.stack(c2_dat[:-2])
x3 = np.stack(c3_dat[:-2])

# Concatenate to form the input training set
col_concat = np.array([x1,x2,x3])
t_col_concat = col_concat.T

```
We also batchify the training set in batches of 32, so each training instance is of shape 32 X 3. Batchifying the input helps us train the model faster.

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

### Preparing the Data Set for gluon RNN
This is very similar to preparing the dataset for unrolled RNN, except for the shape of the input. The dataset should be ordered in the shape (number of example X batch_size). For example, let us consider the sample dataset below:
![Alt text](images/batch3.png?raw=true "batch reshape") <br />
Let try to batch 
In the above image, the input sequence is converted to batch of size 3. By transforming it this way, we loose the temporal relationship between 'O' and 'V', 'M' and 'T' and but we can train in batches (faster).  It is very easy to generate arbitrary length input sequence. During our training, we use a input sequence length of 15. This is a hyperparameter and may require fine tuning for best output.

### Designing RNN in Gluon
Next, we define a class that allows us to create two RNN models we have chosen for our example: GRU (Gated Recurrent Unit)](https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.rnn.GRU) and [LSTM](https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.rnn.LSTM). GRU is a simpler version of LSTM, and performs just as well. You can find a comparison study [here](https://arxiv.org/abs/1412.3555). The models are created with the following Python snippet:


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
The constructor of class creates the neural units that will be used in our forward pass. You can pass type of RNN layer (LSTM, GRU or Vanilla RNN) to you would like to use and run a comparison test among different models.  The forward pass is the method that will be called during our training to generate the loss associated with the training data.

The forward pass function starts by creating an [embedding layer](https://mxnet.incubator.apache.org/api/python/gluon.html#mxnet.gluon.nn.Embedding) for the input character. You can look at our [previous blog post](https://www.oreilly.com/ideas/sentiment-analysis-with-apache-mxnet) for more details on embedding. The output of the embedding layer is provided as input to the RNN. The RNN returns an output as well as hidden state. There is dropout layer  The output produced by the RNN is passed to a decoder (dense unit), which predicts the next character in the neural network and also generates the loss.

We also have a “begin state” function that initializes the initial hidden state of the model.

### Training the neural network

After defining the network. we have to train the neural network for it to learn the underlying distribution.


```python
def trainGluonRNN(epochs,train_data,seq=seq_length):
    for epoch in range(epochs):
        total_L = 0.0
        hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx = context)
        for ibatch, i in enumerate(range(0, train_data.shape[0] - 1, seq_length)):
            data, target = get_batch(train_data, i,seq)
            hidden = detach(hidden)
            with autograd.record():
                output, hidden = model(data, hidden)
                L = loss(output, target) # this is total loss associated with seq_length
                L.backward()

            grads = [i.grad(context) for i in model.collect_params().values()]
            # Here gradient is for the whole batch.
            # So we multiply max_norm by batch_size and seq_length to balance it.
            gluon.utils.clip_global_norm(grads, clip * seq_length * batch_size)

            trainer.step(batch_size)
            total_L += mx.nd.sum(L).asscalar()
```

Each epoch starts by initializing the hidden units to zero. While training each batch, we detach the hidden unit from computational graph so that we don’t backpropagate the gradient beyond the sequence length (15 in our case). If we don’t detach the hidden state, the gradient is passed to the beginning of hidden state (t=0). After detaching, we calculate the loss and use the backward function to back-propagate the loss in order to fine tune the weights. We also normalize the gradient by multiplying it by the sequence length and batch size.

### Text generation

After training for 200 epochs, we can generate random text. The weights of trained model is avaliable [here](https://www.dropbox.com/s/7b1fw94s1em5po0/gluonlstm_2?dl=0). You can download the model parammeters and load it using [model.load_params](https://mxnet.incubator.apache.org/api/python/module/module.html?highlight=load#mxnet.module.BaseModule.load_params) function.

To generate  text, we initialize the hidden state.
```python
 hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx=context)
```
Remember we dont need to reset the hidden state since we dont backpropagte the loss (fine tune the weights).



Then we reshape the input sequence to vector the model accepts using Mxnet arrays. Then we recurrsively :
Reshape the input to the shape the RNN model accepts
```python
 sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T
                                ,ctx=context)
```

generate output char 'c'. We look at the argmax of the output produced by the network

```python
output,hidden = model(sample_input,hidden)
output,hidden = model(sample_input,hidden)
index = mx.nd.argmax(output, axis=1)
index = index.asnumpy()
count = count + 1
```

appending output char 'c' to input string

```python
sample_input = mx.nd.array(np.array([idx[0:seq_length]]).T,ctx=context)
new_string = new_string + indices_char[index[-1]]
input_string = input_string[1:] + indices_char[index[-1]]
```

slice the first charact of input string.

```python
 new_string = new_string + indices_char[index[-1]]
        input_string = input_string[1:] + indices_char[index[-1]]
```


```python
# a nietzsche like text generator
import sys
def generate_random_text(model,input_string,seq_length,batch_size,sentence_length_to_generate):
    count = 0
    new_string = ''
    cp_input_string = input_string
    hidden = model.begin_state(func = mx.nd.zeros, batch_size = batch_size, ctx=context)
    while count < sentence_length_to_generate:
        idx = [char_indices[c] for c in input_string]
        if(len(input_string) != seq_length):
            print(len(input_string))
            raise ValueError('there was a error in the input ')
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

If you look at the text generated, we will note the model has learnt open and close quotations(""). It has definite structure and looks similar to 'nietzsche'.

Next we will look into generative models for images and specially GAN (Generative Adversarial network)

## Generative Adversarial Network (GAN)

[Generative adversarial network](https://arxiv.org/abs/1406.2661) is a neural network model based on a [zero-sum game](https://en.wikipedia.org/wiki/Zero-sum_game) from game theory. The application typically consists of two different neural networks called Discriminator and Generator, where each network tries to outperform the other. Let us consider an example to understand GAN network.

Let’s assume that there is a bank (discriminator) that detects whether a given currency is real or fake using machine learning. A fraudster (generator) builds a machine learning model to counterfeit fake currency notes by looking at the real currency notes, and deposits them in bank. The bank tries to identify the currencies deposited as fake.
![Alt text](images/GAN_SAMPLE.png?raw=true "Generative Adversarial Network")

If the bank tells the fraudster why it classified these notes as fake,  he can improve his model based on those reasons. After multiple iterations, the bank cannot find the difference between the “real” and “fake” currency. This is the basic idea behind GAN. So now let's implement a simple GAN network.

I encourage you to download [the notebook](https://github.com/sookinoby/generative-models/blob/master/GAN.ipynb).
You are welcome to adjust the hyperparameters and experiment with different approaches to neural network architecture.

### Preparing the DataSet

We use a library called [Brine](https://docs.brine.io/getting_started.html) to download our dataset. Brine has many data sets, so we can choose the data set that we want to download. To install Brine and download our data set, do the following:

1. pip install brine-io
2. brine install jayleicn/anime-faces

For this tutorial, I am using the Anime-faces data set, which contains over 100,000 anime images collected from the Internet.

Once the dataset is downloaded, you can load it using the following code:

```python
# brine for loading anime-faces dataset
import brine
anime_train = brine.load_dataset('jayleicn/anime-faces')
```


We also need to normalize the pixel value of each image to [-1 to 1] and reshape each image from (width X height X channels) to (channels X width X height), because the latter format is what MxNet expects. The transform function does the job of reshaping the input image into the required shape expected by the MxNet model.


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
```
The getImageList functions reads the images from the training_folder and returns the images as list, which is then transformed into MxNet array.

```python
# Read images, call the transform function, attach it to list
def getImageList(base_path,training_folder):
    img_list = []
    for train in training_folder:
        fname = base_path + train.image
        img_arr = mx.image.imread(fname)
        img_arr = transform(img_arr, target_wd, target_ht)
        img_list.append(img_arr)
    return img_list
```
base_path = 'brine_datasets/jayleicn/anime-faces/images/'
img_list = getImageList('brine_datasets/jayleicn/anime-faces/images/',training_fold)
```


### Designing the network
We now need to design the two separate networks, the discriminator and the generator. The generator takes a random vector of shape (batchsize X N ), where N is an integer, and converts it to a image of shape (batchsize X channels X width X height). It uses [transpose convolutions](http://deeplearning.net/software/theano_versions/dev/tutorial/conv_arithmetic.html#no-zero-padding-unit-strides-transposed) to upscale the input vectors. This is very similar to how a decoder unit in an [autoencoder](https://en.wikipedia.org/wiki/Autoencoder) maps a lower-dimension vector into a higher-dimensional vector representation. You can choose to design your own generator network, the only the thing you need to be careful about is the input and the output shapes. The input to generator network should be of low dimension (we use 1X150 dimension, latent_z_size) and output should be the expected number of channels (3 , for color images) , width and height (3 x width x height). A snippet of a generator network follows.


```python
# simple generator. Use any models but should upscale the latent variable(randome vectors) to 64 * 64 * 3 channel image
with netG.name_scope():
     # input is random_z (batchsize X 150 X 1), going into a tranposed convolution
    netG.add(nn.Conv2DTranspose(ngf * 8, 4, 1, 0))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # output size. (ngf*8) x 4 x 4
    netG.add(nn.Conv2DTranspose(ngf * 4, 4, 2, 1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # output size. (ngf*8) x 8 x 8
    netG.add(nn.Conv2DTranspose(ngf * 2, 4, 2, 1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # output size. (ngf*8) x 16 x 16
    netG.add(nn.Conv2DTranspose(ngf, 4, 2, 1))
    netG.add(nn.BatchNorm())
    netG.add(nn.Activation('relu'))
    # output size. (ngf*8) x 32 x 32
    netG.add(nn.Conv2DTranspose(nc, 4, 2, 1))
    netG.add(nn.Activation('tanh')) # use tanh , we need an output that is between -1 to 1, not 0 to 1 
    # Rememeber the input image is normalised between -1 to 1, so should be the output
    # output size. (nc) x 64 x 64
```

Our discriminator is a binary image classification network that maps the image of shape (batchsize X channels X width x height) into a lower-dimension vector of shape (batchsize X 1). This is similar to an encoder that converts a higher-dimension image representation into a lower-dimension one. Again, you choose to use any model that does binary classification with reasonable accuraccy. A snippet of the discriminator network follows:

```python
with netD.name_scope():
    # input is (nc) x 64 x 64
    netD.add(nn.Conv2D(ndf, 4, 2, 1))
    netD.add(nn.LeakyReLU(0.2))
    # output size. (ndf) x 32 x 32
    netD.add(nn.Conv2D(ndf * 2, 4, 2, 1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # output size. (ndf) x 16 x 16
    netD.add(nn.Conv2D(ndf * 4, 4, 2, 1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # output size. (ndf) x 8 x 8
    netD.add(nn.Conv2D(ndf * 8, 4, 2, 1))
    netD.add(nn.BatchNorm())
    netD.add(nn.LeakyReLU(0.2))
    # output size. (ndf) x 4 x 4
    netD.add(nn.Conv2D(1, 4, 1, 0))
```

## Training the GAN network
The training of a GAN network is not straightforward, but it is simple. The following diagram illustrates the training process.  ![Alt text](images/GAN_Model.png?raw=true "GAN training") <br />

The real images are given a label of 1 and the fake images are given a label of 0.

```python
#real label is the labels of real image
real_label = nd.ones((batch_size,), ctx=ctx)
#fake labels is label associated with fake image
fake_label = nd.zeros((batch_size,),ctx=ctx)
```
### Training the discriminator

A real image is now passed to the discriminator, to determine if it is real or fake, and the loss associated with the prediction is calculated as errD_real.

 ```python
# train with real image
output = netD(data).reshape((-1, 1))
#The loss is a real valued number
errD_real = loss(output, real_label)
```

In the next step, a random noise random_z is passed to the generator network to produce a random image. This image is then passed to the discriminator to classify it as real (1) or fake(0), thereby producing a loss, errD_fake.

 ```python
#train with fake image, see what the discriminator predicts
#creates fake image
fake = netG(random_z)
# pass it to the discriminator
output = netD(fake.detach()).reshape((-1, 1))
errD_fake = loss(output, fake_label)
 ```

The total error is backpropagated to tune the weights of the discriminator.

 ```python
#compute the total error for fake image and the real image
errD = errD_real + errD_fake
#improve the discriminator skill by backpropagating the error
errD.backward()
```

### Training the generator

The random noise(random_z) vector used in the training of discriminator is used again to generate a fake image. Then we pass the fake image to the discriminator network to obtain the classification output, and loss is calculated. The loss is then used to fine tune the network.

```python
fake = netG(random_z)
output = netD(fake).reshape((-1, 1))
errG = loss(output, real_label)
errG.backward()
```

### Generating new fake images
The model weights are available [here](https://www.dropbox.com/s/uu45cq5y6uigiro/GAN_t2.params?dl=0). You can download the model parammeters and load it using [model.load_params](https://mxnet.incubator.apache.org/api/python/module/module.html?highlight=load#mxnet.module.BaseModule.load_params) function.
We can use the generator network to create new fake images by providing 150 dimension random input to the network. 

 ![Alt text](images/GAN_image.png?raw=true "GAN generated images")<br />

```
#Let’s generate some random images
num_image = 8
for i in range(num_image):
    # random input for generating images
    latent_z = mx.nd.random_normal(0, 1, shape=(1, latent_z_size, 1, 1), ctx=ctx)
    img = netG(random_z)
    plt.subplot(2,4,i+1)
    visualize(img[0])
plt.show()
```


# Conclusion
Generative models open up new opportunities for deep learning. This article has explored some of the popular generative models for text and image data. We learned the basics of RNN and how RNN can be constructed using a feed forward neural network. We also used LSTM/GRU/Vanilla RNN to generate text similar to Friedrich Nietzsche. Finally, we learned about GAN models and generated images similar to input data (Anime Characters). 
