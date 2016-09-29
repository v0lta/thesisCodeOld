import numpy as np
import tensorflow as tf

def target_list_to_sparse_tensor(targetList):
    '''make tensorflow SparseTensor from list of targets, with each element
       in the list being a list or array with the values of the target sequence
       (e.g., the integer values of a character map for an ASR target string)
    '''
    indices = []
    vals = []
    lengths = []
    for tI, target in enumerate(targetList):
        for seqI, val in enumerate(target):
            lengths.append(len(target))
            if val != 0:
                indices.append([tI, seqI])
                vals.append(val)
    shape = [len(targetList), np.max(lengths)]

    return (np.array(indices), np.array(vals), np.array(shape))

def data_lists_to_batches(inputList, targetList, batchSize, maxSteps=None):
    '''Takes a list of input matrices and a list of target arrays and returns
       a list of batches, with each batch being a 3-element tuple of inputs,
       targets, and sequence lengths.
       inputList: list of 2-d numpy arrays with dimensions nFeatures x timesteps
       targetList: list of 1-d arrays or lists of ints
       batchSize: int indicating number of inputs/targets per batch
       returns: dataBatches: list of batch data tuples,
                where each batch tuple (inputs, targets, seqLengths)
                consists of:
                    inputs  = 3-d array w/ shape nTimeSteps x batchSize x nFeatures
                    targets = tuple required as input for SparseTensor
                    seqLengths = 1-d array with int number of timesteps for
                                 each sample in batch
                maxSteps: maximum number of time steps across all samples'''
    
    assert len(inputList) == len(targetList)
    nFeatures = inputList[0].shape[0]
    
    if maxSteps == None:
        maxSteps = 0
        for inp in inputList:
            maxSteps = max(maxSteps, inp.shape[1])
    
    #randIxs = np.random.permutation(len(inputList)) #randomly mix the batches.
    batchIxs = np.asarray(range(len(inputList))) #do not randomly permutate.
    start, end = (0, batchSize)
    dataBatches = []

    while end <= len(inputList):
        batchSeqLengths = np.zeros(batchSize)
        for batchI, origI in enumerate(batchIxs[start:end]):
            batchSeqLengths[batchI] = inputList[origI].shape[-1]
        batchInputs = np.zeros((maxSteps, batchSize, nFeatures))
        batchTargetList = []
        for batchI, origI in enumerate(batchIxs[start:end]):
            padSecs = maxSteps - inputList[origI].shape[1]
            batchInputs[:,batchI,:] = np.pad(normalizeInput(inputList[origI].T),
                                            ((0,padSecs),(0,0)),
                                             'constant', constant_values=0)
            batchTargetList.append(targetList[origI])
        dataBatches.append((batchInputs, target_list_to_sparse_tensor(batchTargetList),
                          batchSeqLengths))
        start += batchSize
        end += batchSize
        
    return (dataBatches, maxSteps)
            
def normalizeInput(inputArray):
    zeroMean = inputArray - np.mean(inputArray)
    stdOneZeroMean = zeroMean/np.std(zeroMean)
    return stdOneZeroMean
    
