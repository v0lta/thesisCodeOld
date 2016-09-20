import tensorflow as tf
from tensorflow.python.ops import ctc_ops as ctc
from tensorflow.python.ops import rnn_cell
from tensorflow.python.ops.rnn import bidirectional_rnn
import numpy as np
from prepareTimit import load_batched_timit
from prepareTimit39 import load_batched_timit39
import matplotlib.pyplot as plt

####Learning Parameters
learningRate = 0.0001       #too low?
momentum = 0.9              #play with this.
nEpochs = 4
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

meanLst = []
stdLst = []
for i in range(0,batchedData[0][0][:,:,:].shape[1]):
    meanLst.append(np.mean(batchedData[0][0][:,i,:]))
    stdLst.append(np.std(batchedData[0][0][:,i,:]))
    
graph = tf.Graph()
with graph.as_default():
    inputX = tf.placeholder(tf.float32, shape=(maxTimeSteps, batchSize, nFeatures))
    print(tf.Tensor.get_shape(inputX))
    #turn into list [maxTimeSteps],[BatchSize,nFeatures]
    inputList = tf.unpack(inputX,num = maxTimeSteps,axis = 0)
    print(len(inputList))


batchInputs, batchTargetSparse, batchSeqLengths = batchedData[0]    

with tf.Session(graph=graph) as session:
    lst = session.run([inputList], feed_dict = {inputX : batchInputs})    

print(type(lst))
print(type(lst[0]))
print(type(lst[0][0]))
print(len(lst[0]))
print(lst[0][0].shape)