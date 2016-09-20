'''
Example of a single-layer bidirectional long short-term memory network trained with
connectionist temporal classification to predict phoneme sequences from nFeatures x nFrames
arrays of Mel-Frequency Cepstral Coefficients.  This is basically a recreation of an experiment
on the TIMIT data set from chapter 7 of Alex Graves's book (Graves, Alex. Supervised Sequence 
Labelling with Recurrent Neural Networks, volume 385 of Studies in Computational Intelligence.
Springer, 2012.), minus the early stopping.

Authors: mortiz Wolter and Jon Rein 
'''

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
from prepareTimit39 import load_batched_timit39

####Learning Parameters
learningRate = 0.0001
momentum = 0.9
#learningRate = 0.0001       #too low?
#momentum = 0.6              #play with this.
nEpochs = 20
batchCount = 8          #too small??
batchCount_val = 1
batchCount_test = 1

####Network Parameters
nFeatures = 40
nHidden = 128
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
    feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals, targetShape: batchTargetShape, seqLengths: batchSeqLengths}
    return feedDict, batchSeqLengths

def blstm(inputList, weightsBLSTM, biasesBLSTM):
    initializer = tf.random_normal_initializer(0.0,0.1)
    #initializer = tf.random_normal_initializer(0.0,np.sqrt(2.0 / (2*nHidden)))
    #initializer = None
    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes = True, state_is_tuple = True,
                                    initializer = initializer)
    backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes = True, state_is_tuple = True,
                                     initializer = initializer)
    #compute the bidirectional RNN output throw away the states.
    #the output is a length T list consiting of ([time][batch][cell_fw.output_size + cell_bw.output_size]) tensors.
    listH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype = tf.float32,
                                       scope = 'BDLSTM_H1')
                                       
    logits = [tf.matmul(T, weightsBLSTM) + biasesBLSTM for T in listH1]
    
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
    inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, None, nFeatures))
    #Prep input data to fit requirements of rnn.bidirectional_rnn
    #Split to get a list of 'n_steps' tensors of shape (batch_size, nFeatures)
    inputList = tf.unpack(inputX,num = maxTimeSteps,axis = 0)
    #Target indices, values and shape used to create a sparse tensor.
    targetIxs = tf.placeholder(tf.int64, shape=None)    #indices
    targetVals = tf.placeholder(tf.int32, shape=None)   #vals
    targetShape = tf.placeholder(tf.int64, shape=None)  #shape
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=None)
    
    #### Weights & biases
    #weightsBLSTM = tf.Variable(tf.random_normal([nHidden*2, nClasses], mean=0.0,
    #                        stddev=0.1, dtype=tf.float32, seed=None, name=None))
    weightsBLSTM = tf.Variable(tf.truncated_normal([nHidden*2, nClasses],
                                                   stddev=np.sqrt(2.0 / (2*nHidden))))
    biasesBLSTM = tf.Variable(tf.zeros([nClasses]))

    #### Network
    noisyInputs = [tf.random_normal(tf.shape(T),0.0,0.6) + T for T in inputList]
    logits = blstm(noisyInputs, weightsBLSTM, biasesBLSTM)
    
    #### Optimizing
    # logits3d (maxTimeSteps, batchSize, nClasses), pack puts the list into a big matrix.
    logits3d = tf.pack(logits)
    print("logits 3d shape:", tf.Tensor.get_shape(logits3d))
    loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
    optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)

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
epoch_error_lst = []
epoch_error_lst_val = []
with tf.Session(graph=graph) as session:
    print('Initializing')
    tf.initialize_all_variables().run()
    
    #check untrained performance.
    batchErrors = np.zeros(len(batchedData))
    batchRandIxs = np.array(range(0,len(batchedData)))
    for batch, batchOrigI in enumerate(batchRandIxs):
        feedDict, batchSeqLengths = createDict(batchedData[batchOrigI]) 
        l, er, lmt = session.run([loss, errorRate, logitsMaxTest], feed_dict=feedDict)
        print(np.unique(lmt)) #print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
        if (batch % 1) == 0:
            print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
            print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
        batchErrors[batch] = er*len(batchSeqLengths)
    epochErrorRate = batchErrors.sum() / totalN
    epoch_error_lst.append(epochErrorRate)
    print('Untrained error rate:', epochErrorRate)
    
    feedDict, _ = createDict(batchedData_val[0]) 
    vl, ver = session.run([loss, errorRate], feed_dict=feedDict)
    print("untrained validation loss: ", vl, " validation error rate", ver )
    epoch_error_lst_val.append(ver)
        
    for epoch in range(nEpochs):
        print('Epoch', epoch+1, '...')
        batchErrors = np.zeros(len(batchedData))
        batchRandIxs = np.random.permutation(len(batchedData)) #randomize batch order
        for batch, batchOrigI in enumerate(batchRandIxs):
            feedDict, batchSeqLengths = createDict(batchedData[batchOrigI]) 
            _, l, er, lmt = session.run([optimizer, loss, errorRate, logitsMaxTest], feed_dict=feedDict)
            print(np.unique(lmt)) #print unique argmax values of first sample in batch; should be blank for a while, then spit out target values
            if (batch % 1) == 0:
                print('Minibatch', batch, '/', batchOrigI, 'loss:', l)
                print('Minibatch', batch, '/', batchOrigI, 'error rate:', er)
            batchErrors[batch] = er*len(batchSeqLengths)
        epochErrorRate = batchErrors.sum() / totalN
        epoch_error_lst.append(epochErrorRate)
        print('Epoch', epoch+1, 'error rate:', epochErrorRate)
        #compute the validation error
        feedDict, _ = createDict(batchedData_val[0]) 
        vl, ver = session.run([loss, errorRate], feed_dict=feedDict)
        print("validation loss: ", vl, " validation error rate", ver )
        epoch_error_lst_val.append(ver)
        print("validation errors", epoch_error_lst_val )
    #run the network on the test data set.
    feedDict, _ = createDict(batchedData_test[0]) 
    tl, ter = session.run([loss, errorRate], feed_dict=feedDict)
    print("test loss: ", tl, " test error rate", ter )
    
    

import matplotlib.pyplot as plt
plt.plot(np.array(epoch_error_lst))
plt.plot(np.array(epoch_error_lst_val))
plt.show()

