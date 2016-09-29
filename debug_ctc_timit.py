
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
from prepareTimit import load_batched_timit

####Learning Parameters
learningRate = 0.0001
momentum = 0.9
#learningRate = 0.0001       #too low?
#momentum = 0.9              #play with this.
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
data_batches = load_batched_timit(batchCount, batchCount_val, batchCount_test)
batchedData, maxTimeSteps, totalN, batchSize = data_batches[0]
batchedData_val, maxTimeSteps_val, totalN_val, batchSize_val = data_batches[1]
batchedData_test, maxTimeSteps_test, totalN_test, batchSize_test = data_batches[2]

#check if the padding has been done right.
assert(maxTimeSteps == maxTimeSteps_val)
assert(maxTimeSteps == maxTimeSteps_test)

    
#from IPython.core.debugger import Tracer
#Tracer()() 

def createDict(batchedData, noise = False):
    batchInputs, batchTargetSparse, batchSeqLengths = batchedData
    batchTargetIxs, batchTargetVals, batchTargetShape = batchTargetSparse
    feedDict = {inputX: batchInputs, targetIxs: batchTargetIxs, targetVals: batchTargetVals,
                targetShape: batchTargetShape, seqLengths: batchSeqLengths, inputNoise: noise}
    return feedDict, batchSeqLengths

def blstm(inputList, weightsBLSTM, biasesBLSTM):
    initializer = tf.random_normal_initializer(0.0,0.1)
    forwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes = True, state_is_tuple = True,
                                    initializer = initializer)
    backwardH1 = rnn_cell.LSTMCell(nHidden, use_peepholes = True, state_is_tuple = True,
                                     initializer = initializer)
    #compute the bidirectional RNN output throw away the states.
    #the output is a length T list consiting of ([time][batch][cell_fw.output_size + cell_bw.output_size]) tensors.
    
    listH1, _, _ = bidirectional_rnn(forwardH1, backwardH1, inputList, dtype = tf.float32,
                                       scope = 'BDLSTM_H1')
                                       
    logits = [tf.matmul(T, weightsBLSTM) + biasesBLSTM for T in listH1]
    #logits [time][batchSize, 40]
    print("length logit list:", len(logits))
    print("logit list element shape:", tf.Tensor.get_shape(logits[0]))
    return logits

####Define graph
print('Defining graph')
graph = tf.Graph()
with graph.as_default():
    inputNoise = tf.placeholder(tf.bool, shape=())
    
    #### Graph input shape=(maxTimeSteps, batchSize, nFeatures),  but the first two change.
    inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, None, nFeatures))
    #Prep input data to fit requirements of rnn.bidirectional_rnn
    #  Reshape to 2-D tensor (nTimeSteps*batchSize, nfeatures)
    inputXrs = tf.reshape(inputX, [-1, nFeatures])
    #  Split to get a list of 'n_steps' tensors of shape (batch_size, n_hidden)
    inputList = tf.split(0, maxTimeSteps, inputXrs)
    #Target indices, values and shape used to create a sparse tensor.
    targetIxs = tf.placeholder(tf.int64, shape=None)    #indices
    targetVals = tf.placeholder(tf.int32, shape=None)   #vals
    targetShape = tf.placeholder(tf.int64, shape=None)  #shape
    targetY = tf.SparseTensor(targetIxs, targetVals, targetShape)
    seqLengths = tf.placeholder(tf.int32, shape=None)
    
    #### Weights & biases
    weightsBLSTM = tf.Variable(tf.random_normal([nHidden*2, nClasses], mean=0.0,
                            stddev=0.1, dtype=tf.float32, seed=None, name=None))
    #weightsBLSTM = tf.Variable(tf.truncated_normal([nHidden*2, nClasses],
    #                                               stddev=np.sqrt(2.0 / (2*nHidden))))
    biasesBLSTM = tf.Variable(tf.zeros([nClasses]))

    #### Network
    noisyInputs = [tf.random_normal(tf.shape(T),0.0,0.7) + T for T in inputList]
    logits = blstm(noisyInputs, weightsBLSTM, biasesBLSTM)
    
    #### Optimizing
    # logits3d (maxTimeSteps, batchSize, nClasses), pack puts the list into a big matrix.
    logits3d = tf.pack(logits)
    loss = tf.reduce_mean(ctc.ctc_loss(logits3d, targetY, seqLengths))
    optimizer = tf.train.MomentumOptimizer(learningRate, momentum).minimize(loss)

    #### Evaluating
    logitsMaxTest = tf.slice(tf.argmax(logits3d, 2), [0, 0], [seqLengths[0], 1])
    #predictions = tf.to_int32(ctc.ctc_beam_search_decoder(logits3d, seqLengths)[0][0])
    predictions = tf.to_int32(ctc.ctc_greedy_decoder(logits3d, seqLengths)[0][0])
    errorRate = tf.reduce_mean(tf.edit_distance(predictions, targetY, normalize=True)) 
    
#from IPython.core.debugger import Tracer
#Tracer()()
    
    
with tf.Session(graph=graph) as session:
    tf.initialize_all_variables().run()
    feedDict, _ = createDict(batchedData_test[0]) 
    tl, ter, pred, tY = session.run([loss, errorRate, predictions, targetY], feed_dict=feedDict)
