

# ### Generating names with recurrent neural networks


import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
get_ipython().magic('matplotlib inline')


# # Our data
# The dataset contains ~8k earthling names from different cultures, all in latin transcript.
# 
# This notebook has been designed so as to allow you to quickly swap names for something similar: deep learning article titles, IKEA furniture, pokemon names, etc.



import os
start_token = " "

with open("names") as f:
    names = f.read()[:-1].split('\n')
    names = [start_token+name for name in names]



print ('n samples = ',len(names))
for x in names[::1000]:
    print (x)
    



MAX_LENGTH = max(map(len,names))
print("max length =", MAX_LENGTH)

plt.title('Sequence length distribution')
plt.hist(list(map(len,names)),bins=25);


# # Text processing
# 
# First we need next to collect a "vocabulary" of all unique tokens i.e. unique characters. We can then encode inputs as a sequence of character ids.

#all unique characters go here
tokens = set()

for name in names:
    tokens|=set(name)
    
tokens = list(tokens)

n_tokens = len(tokens)
print ('n_tokens = ',n_tokens)

assert 50 < n_tokens < 60


# ### Cast everything from symbols into identifiers
# 
# Tensorflow string manipulation is a bit tricky, so we'll work around it. 
# We'll feed our recurrent neural network with ids of characters from our dictionary.
# 
# To create such dictionary, let's assign 



token_to_id = {item:idx for idx, item in enumerate(tokens)}
print (token_to_id)




assert len(tokens) == len(token_to_id), "dictionaries must have same size"

for i in range(n_tokens):
    assert token_to_id[tokens[i]] == i, "token identifier must be it's position in tokens list"

print("Seems alright!")




def to_matrix(names,max_len=None,pad=0,dtype='int32'):
    """Casts a list of names into rnn-digestable matrix"""
    
    max_len = max_len or max(map(len,names))
    names_ix = np.zeros([len(names),max_len],dtype) + pad

    for i in range(len(names)):
        name_ix = list(map(token_to_id.get,names[i]))
        names_ix[i,:len(name_ix)] = name_ix

    return names_ix.T




#Example: cast 4 random names to matrices, pad with zeros
print('\n'.join(names[::2000]))
print(to_matrix(names[::2000]).T)


# # Recurrent neural network
# 
# We can rewrite recurrent neural network as a consecutive application of dense layer to input x_t and previous rnn state h_t. 


import keras
from keras.layers import Concatenate,Dense,Embedding

rnn_num_units = 64
embedding_size = 16

#Let's create layers for our recurrent network
#Note: we create layers but we don't "apply" them yet

embed_x = Embedding(n_tokens,embedding_size) # an embedding layer that converts character ids into embeddings
#one hot-encoding size n_tokens * embeding_size

#a dense layer that maps input and previous state to new hidden state, [x_t,h_t]->h_t+1
get_h_next = Dense(units=rnn_num_units, activation = 'tanh')
#create next layer input 

#a dense layer that maps current hidden state to probabilities of characters [h_t+1]->P(x_t+1|h_t+1)
get_probas = Dense(units=n_tokens, activation='softmax')
#generate probabilities of the next layer, in order to get the new future input 

#Note: please either set the correct activation to Dense or write it manually in rnn_one_step




def rnn_one_step(x_t, h_t):
    """
    Recurrent neural network step that produces next state and output
    given prev input and previous state.
    We'll call this method repeatedly to produce the whole sequence.
    
    Follow inline isntructions to complete the function.
    """
    #convert character id into embedding
    x_t_emb = embed_x(tf.reshape(x_t,[-1,1]))[:,0]
    
    #concatenate x embedding and previous h state
    x_and_h = Concatenate()([x_t_emb,h_t])
    
    #compute next state given x_and_h
    h_next = get_h_next(x_and_h)
    
    #get probabilities for language model P(x_next|h_next)
    output_probas = get_probas(h_next)
    
    return output_probas,h_next


# ### RNN loop
# 
# Once rnn_one_step is ready, let's apply it in a loop over name characters to get predictions.
# 
# Let's assume that all names are at most length-16 for now, so we can simply iterate over them in a for loop.
# 


input_sequence = tf.placeholder('int32',(MAX_LENGTH,None))
batch_size = tf.shape(input_sequence)[1]

predicted_probas = []
h_prev = tf.zeros([batch_size,rnn_num_units]) #initial hidden state

for t in range(MAX_LENGTH):
    x_t = input_sequence[t]
    probas_next,h_next = rnn_one_step(x_t,h_prev)    
    h_prev = h_next
    predicted_probas.append(probas_next)
    
predicted_probas = tf.stack(predicted_probas)


# ## RNN: loss and gradients
# 
# Let's gather a matrix of predictions for $P(x_{next}|h)$ and the corresponding correct answers.
# 
# Our network can then be trained by minimizing crossentropy between predicted probabilities and those answers.



predictions_matrix = predicted_probas[:-1]
print(predictions_matrix)
answers_matrix = tf.one_hot(input_sequence[1:], n_tokens)
print(answers_matrix)



loss = tf.reduce_mean(keras.losses.categorical_crossentropy(answers_matrix, predictions_matrix))

optimize = tf.train.AdamOptimizer().minimize(loss)


# ### The training loop


from IPython.display import clear_output
from random import sample
s = keras.backend.get_session()
s.run(tf.global_variables_initializer())
history = []



for i in range(1000):
    
    batch = to_matrix(sample(names,32),max_len=MAX_LENGTH)
    loss_i,_ = s.run([loss,optimize],{input_sequence:batch})
    
    
    history.append(loss_i)
    if (i+1)%100==0:
        clear_output(True)
        plt.plot(history,label='loss')
        plt.legend()
        plt.show()

assert np.mean(history[:10]) > np.mean(history[-10:]), "RNN didn't converge."


# ### RNN: sampling
# Once we've trained our network a bit, let's get to actually generating stuff. All we need is the `rnn_one_step` function you have written above.



x_t = tf.placeholder('int32',(1,))
h_t = tf.Variable(np.zeros([1,rnn_num_units],'float32'))

next_probs,next_h = rnn_one_step(x_t,h_t)




def generate_sample(seed_phrase=' ',max_length=MAX_LENGTH):
    '''
    The function generates text given a phrase of length at least SEQ_LENGTH.
        
    parameters:
        The phrase is set using the variable seed_phrase
        The optional input "N" is used to set the number of characters of text to predict.     
    '''
    x_sequence = [token_to_id[token] for token in seed_phrase]
    s.run(tf.assign(h_t,h_t.initial_value))
    
    #feed the seed phrase, if any
    for ix in x_sequence[:-1]:
         s.run(tf.assign(h_t,next_h),{x_t:[ix]})
    
    #start generating
    for _ in range(max_length-len(seed_phrase)):
        x_probs,_ = s.run([next_probs,tf.assign(h_t,next_h)],{x_t:[x_sequence[-1]]})
        x_sequence.append(np.random.choice(n_tokens,p=x_probs[0]))
        
    return ''.join([tokens[ix] for ix in x_sequence])



for _ in range(10):
    print(generate_sample())



for _ in range(50):
    print(generate_sample(' Trump'))




