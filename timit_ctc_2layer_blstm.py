'''
Example of a two-layer bidirectional long short-term memory network trained with
connectionist temporal classification to predict phoneme sequences from nFeatures x nFrames
arrays of Mel-Frequency Cepstral Coefficients.

Author: mortiz Wolter
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from math import pow

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
from prepare_timit39 import load_batched_timit39

#to store the data and name it properly.
import pickle
import socket

####Learning Parameters
learningRate = 0.001
momentum = 0.9
omega = pow(10, -10) #weight regularization term.
inputNoiseStd = 0.65
#learningRate = 0.0001       #too low?
#momentum = 0.6              #play with this.
#nEpochs = 240
batchCount = 8          #too small??
batchCount_val = 1
batchCount_test = 1

####Network Parameters
nFeatures = 40
nHidden = 64
nClasses = 40 #39 phonemes, plus the "blank" for CTC

####Load data
print('Loading data')
data_batches = load_batched_timit39(batchCount, batchCount_val, batchCount_test)
batchedData, maxTimeSteps, totalN, batchSize = data_batches[0]
batchedData_val, maxTimeSteps_val, totalN_val, batchSize_val = data_batches[1]
batchedData_test, maxTimeSteps_test, totalN_test, batchSize_test = data_batches[2]

#check if the padding has been done right.
assert(maxTimeSteps == maxTimeSteps_val)
assert(maxTimeSteps == maxTimeSteps_test)

    
#from IPython.core.debugger import Tracer
#Tracer()()

def createDict(batchedData):
    batchInputs, batchTargetSparse, batchSeqLengths = batchedData
    batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
    feed_dict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals, targetShape: batchTargetShape, seqLengths: batchSeqLengths}
    return feed_dict, batchSeqLengths

def blstm(inputList, hidden_weights, hidden_bias, logit_weights, logit_bias):
    initializer = tf.random_normal_initializer(0.0,0.1)
    #initializer = tf.random_normal_initializer(0.0,np.sqrt(2.0 / (2*nHidden)))
    #initializer = None
    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes = True, state_is_tuple = True,
                                    initializer = initializer)
    backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes = True, state_is_tuple = True,
                                     initializer = initializer)
    forwardH2 = rnn_cell.LSTMCell(nHidden, use_peepholes = True, state_is_tuple = True,
                                     initializer = initializer)
    backwardH2 = rnn_cell.LSTMCell(nHidden, use_peepholes = True, state_is_tuple = True,
                                      initializer = initializer)
    #compute the bidirectional RNN output throw away the states.
    #the output is a length T list consiting of ([time][batch][cell_fw.output_size + cell_bw.output_size]) tensors.
    listH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype = tf.float32,
                                       scope = 'BDLSTM_H1')
    inH2 = [tf.matmul(T, hidden_weights) + hidden_bias for T in listH1]
                                       
    listH2, _, _ = bidirectional_rnn(forwardH2, backwardH2, inH2, dtype = tf.float32,
                                       scope = 'BDLSTM_H2')
    logits = [tf.matmul(T, logit_weights, name = 'logitMatmul') + logit_bias for T in listH2]
    
    print("length logit list:", len(logits))
    print("logit list element shape:", tf.Tensor.get_shape(logits[0]))
    #logits = [tf.nn.softmax(tf.matmul(T, weightsBLSTM) + biasesBLSTM) for T in listH1]
    #logits = [tf.nn.softmax(T) for T in logits]
    return logits

####Define graph
print('Defining graph')
graph = tf.Graph()
with graph.as_default():

    #### Graph input shape=(maxTimeSteps, batchSize, nFeatures),  but the first two change.
    inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, None, nFeatures), name = 'inputFeatures')
    #Prep input data to fit requirements of rnn.bidirectional_rnn
    #Split to get a list of 'n_steps' tensors of shape (batch_size, nFeatures)
    inputList = tf.unpack(inputX,num = maxTimeSteps,axis = 0, name = 'inputList')
    #Target indices, values and shape used to create a sparse tensor.
    targetIxs = tf.placeholder(tf.int64, shape=None)    #indices
    targetVals = tf.placeholder(tf.int32, shape=None)   #vals
    targetShape = tf.placeholder(tf.int64, shape=None)  #shape
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=None, name='seqenceLengths')
    
    #### Weights & biases
    #weightsBLSTM = tf.Variable(tf.random_normal([nHidden*2, nClasses], mean=0.0,
    #                        stddev=0.1, dtype=tf.float32, seed=None, name=None))
    hidden_weights = tf.Variable(tf.truncated_normal([nHidden*2, nHidden],
                                                   mean=0.0, stddev=0.1), name = 'hiddenWeights')
    hidden_bias = tf.Variable(tf.zeros([nHidden]), name = 'hiddenBias')

    logit_weights = tf.Variable(tf.truncated_normal([nHidden*2, nClasses],
                                                   mean=0.0, stddev=0.1), name = 'logitWeights')
    logit_bias = tf.Variable(tf.zeros([nClasses]), name = 'logitBias')


    #### Network
    noisyInputs = [tf.random_normal(tf.shape(T),0.0,inputNoiseStd, name = 'inputNoise') + T for T in inputList]
    logits = blstm(noisyInputs, hidden_weights, hidden_bias,logit_weights, logit_bias )
    
    #### Optimizing
    # logits3d (maxTimeSteps, batchSize, nClasses), pack puts the list into a big matrix.

    trainable_weights = tf.trainable_variables()
    weight_loss = 0
    for trainable in trainable_weights:
            weight_loss += tf.nn.l2_loss(trainable)
    
    logits3d = tf.pack(logits)
    print("logits 3d shape:", tf.Tensor.get_shape(logits3d))
    loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths)) + omega*weight_loss
    uncappedOptimizer = tf.train.MomentumOptimizer(learningRate, momentum)#.minimize(loss)
    #optimizer = tf.train.AdamOptimizer(learningRate).minimize(loss)

    #gradient clipping:
    gvs = uncappedOptimizer.compute_gradients(loss)
    capped_gvs = [(tf.clip_by_value(grad, -1., 1.), var) for grad, var in gvs]
    optimizer = uncappedOptimizer.apply_gradients(capped_gvs)


    #### Evaluating
    logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seqLengths[0], 1])
    #predictions = ctc.ctc_beam_search_decoder(logits3d, seqLengths, beam_width = 100)
    predictions = ctc.ctc_greedy_decoder(logits3d, seqLengths)
    print("predictions", type(predictions))
    print("predictions[0]", type(predictions[0]))
    print("len(predictions[0])", len(predictions[0]))
    print("predictions[0][0]", type(predictions[0][0]))
    hypothesis = tf.to_int32(predictions[0][0])
    
    errorRate = tf.reduce_mean(tf.edit_distance(hypothesis, targetY, normalize=True))
    
#from IPython.core.debugger import Tracer
#Tracer()()
    
####Run session
restarts = 0
epoch_loss_lst = []
epoch_error_lst = []
epoch_error_lst_val = []

with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()

    #check untrained performance.
    batch_loss = np.zeros(len(batchedData))
    batch_errors = np.zeros(len(batchedData))
    batchRandIxs = np.array(range(0,len(batchedData)))
    for batch, batchOrigI in enumerate(batchRandIxs):
        feed_dict, batchSeqLengths = createDict(batchedData[batchOrigI])
        l, wl, er, lmt = session.run([loss, weight_loss, errorRate, logitsMaxTest], feed_dict=feed_dict)
        print(np.unique(lmt)) #print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
        if (batch % 1) == 0:
            print('Minibatch', batch, '/', batchOrigI, 'loss:', l, "weight loss:", wl)
            print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
        batch_errors[batch] = er*len(batchSeqLengths)
        batch_loss[batch] = l*len(batchSeqLengths)
    epochErrorRate = batch_errors.sum() / totalN
    epoch_error_lst.append(epochErrorRate)
    epoch_loss_lst.append(l.sum()/totalN)
    print('Untrained error rate:', epochErrorRate)

    feed_dict, _ = createDict(batchedData_val[0])
    vl, ver = session.run([loss, errorRate], feed_dict=feed_dict)
    print("untrained validation loss: ", vl, " validation error rate", ver )
    epoch_error_lst_val.append(ver)
    
    continue_training = True
    while continue_training:
        epoch = len(epoch_error_lst_val)
        print("params:", learningRate, momentum, omega, inputNoiseStd  )
        print('Epoch', epoch, '...')
        batch_errors = np.zeros(len(batchedData))
        batchRandIxs = np.random.permutation(len(batchedData)) #randomize batch order
        for batch, batchOrigI in enumerate(batchRandIxs):
            feed_dict, batchSeqLengths = createDict(batchedData[batchOrigI])
            _, l, wl, er, lmt = session.run([optimizer, loss, weight_loss, errorRate, logitsMaxTest], feed_dict=feed_dict)
            print(np.unique(lmt)) #print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
            if (batch % 1) == 0:
                print('Minibatch', batch, '/', batchOrigI, 'loss:', l, "weight Loss", wl)
                print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            batch_errors[batch] = er*len(batchSeqLengths)
            batch_loss[batch] = l*len(batchSeqLengths)
        epochErrorRate = batch_errors.sum() / totalN
        epoch_error_lst.append(epochErrorRate)
        epoch_loss_lst.append(l.sum()/totalN)
        print('Epoch', epoch, 'error rate:', epochErrorRate)
        #compute the validation error
        feed_dict, _ = createDict(batchedData_val[0])
        vl, ver, vwl = session.run([loss, errorRate, weight_loss], feed_dict=feed_dict)
        print("vl: ", vl, " ver: ", "vwl: ", vwl)
        epoch_error_lst_val.append(ver)
        print("validation errors", epoch_error_lst_val )
        
        #stop if in the last 50 epochs no progress has been made.
        if epoch > 60:
            min_last_50 = min(epoch_error_lst_val[(epoch-50):epoch])
            min_since_start = min(epoch_error_lst_val[0:(epoch-50)])
            if min_last_50 + 0.001 > (min_since_start):
                continue_training = False
                print("stopping the training.")
        
        if epoch > 600:
                continue_training = False
                
    #run the network on the test data set.
    feed_dict, _ = createDict(batchedData_test[0])
    tl, ter = session.run([loss, errorRate], feed_dict=feed_dict)
    print("test loss: ", tl, " test error rate", ter )

filename = "saved/savedVals2BLSTM." + socket.gethostname() + ".pkl"
pickle.dump([epoch_loss_lst, epoch_error_lst,
             epoch_error_lst_val, ter] , open( filename, "wb" ) )
print("plot values saved at: " + filename)

import matplotlib.pyplot as plt
plt.plot(np.array(epoch_loss_lst)/100.0)
plt.plot(np.array(epoch_error_lst))
plt.plot(np.array(epoch_error_lst_val))
plt.show()
    
