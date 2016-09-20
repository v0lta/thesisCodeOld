import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from prepareTimit import load_batched_timit    
from prepareTimit39 import load_batched_timit39 

def sparseToDense(indices, values, shape):
    dense_array = np.zeros(shape)
    
    for i,index in enumerate(indices):
        dense_array[index[0],index[1]] = values[i]
    return dense_array

def denseToSparse(denseMatrix):
    shape = denseMatrix.shape
    indices = []
    values = []
    for row in range(0,shape[0]):
        for col in range(0,shape[1]):
            if denseMatrix[row,col] != 0:
                indices.append([row,col])
                values.append(denseMatrix[row,col])
    return np.array(indices), np.array(values), np.array(shape)


def arrayListToDense(arrayList, shape):
    dense_array = np.zeros(shape)
    for i,array in enumerate(arrayList):
        dense_array[i,0:len(array)] = array
    return dense_array
    
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
    
####Load data
batchCount = 8
batchCount_val = 1
batchCount_test = 1

print('Loading data')
data_batches = load_batched_timit39(batchCount, batchCount_val, batchCount_test, debug = True)
batchedData, maxTimeSteps, totalN, batchSize = data_batches[0]
batchedData_val, maxTimeSteps_val, totalN_val, batchSize_val = data_batches[1]
batchedData_test, maxTimeSteps_test, totalN_test, batchSize_test = data_batches[2]
target_list_train, target_list_val, target_list_test = data_batches[3]

s1 = len(target_list_test)
lengths = []
for array in target_list_test:
    lengths.append(len(array))
s2 = np.max(lengths)
test_shape = np.array([s1, s2])

batch_ind = batchedData_test[0][1][0]
batch_val = batchedData_test[0][1][1]
batch_shape = batchedData_test[0][1][2]
dense_target_batch = sparseToDense(batch_ind, batch_val, batch_shape)

#get the reference array.
dense_target_array = arrayListToDense(target_list_test, test_shape)

sparse_test = target_list_to_sparse_tensor(target_list_test)
fun_ind = sparse_test[0]
fun_vals = sparse_test[1]
fun_size = sparse_test[2]
dense_target_fun = sparseToDense(fun_ind, fun_vals, fun_size)

#### new code to do the same thing.
check_sparse = denseToSparse(dense_target_array)
check_ind   = check_sparse[0]
check_vals  = check_sparse[1]
check_shape = check_sparse[2]

#convert back to dense array:
dense_target_array_2 = sparseToDense(check_ind, check_vals, check_shape)

print("checking sparse indices:")
print("length:", (len(batch_ind) == len(fun_ind)),
             (len(batch_ind) == len(check_ind)),
             (len(fun_ind) == len(check_ind) ))
print("content:", np.sum( (batch_ind != fun_ind)))


print("checking sparse values:")
print("length:", (len(batch_val) == len(fun_vals)),
             (len(batch_val) == len(check_vals)),
             (len(fun_vals) == len(check_vals) ))
print("content:", np.sum( (batch_val != fun_vals)))


print("target_list_to_sparse_tensor reconstruction error: ", np.sum(dense_target_array != dense_target_fun))
print("batch: ", np.sum(dense_target_array != dense_target_batch))

#print(max(lengths))
plt.matshow(dense_target_array)
plt.matshow(dense_target_fun)
plt.matshow(dense_target_batch)
plt.show()


batch_ind = batchedData[0][1][0]
batch_val = batchedData[0][1][1]
batch_shape = batchedData[0][1][2]
dense_target_batch_train = sparseToDense(batch_ind, batch_val, batch_shape)


def test_edit_distance(truthMat,hypMat):
    graph = tf.Graph()
    
    tmIdx = truthMat[0]
    tmVal = truthMat[1]
    tmShape = truthMat[2]
    
    hmIdx = hypMat[0]
    hmVal = hypMat[1]
    hmShape = hypMat[2]
    
    #from IPython.core.debugger import Tracer
    #Tracer()()
    
    with graph.as_default():
        truth = tf.SparseTensor(tmIdx, tmVal, tmShape)
        hyp = tf.SparseTensor(hmIdx, hmVal, hmShape)
        editDist = tf.edit_distance(hyp, truth, normalize=True)
        errorRate = tf.reduce_mean(editDist)

    with tf.Session(graph=graph) as session:
        errorRate, dist = session.run([errorRate, editDist])
        print("Edit distance", dist)
        print("Error Rate", errorRate)    

def run_edit_test():
    ####Load data
    batchCount = 8
    batchCount_val = 1
    batchCount_test = 1

    print('Loading data')
    data_batches = load_batched_timit(batchCount, batchCount_val, batchCount_test)
    batchedData, maxTimeSteps, totalN, batchSize = data_batches[0]
    batchedData_val, maxTimeSteps_val, totalN_val, batchSize_val = data_batches[1]
    batchedData_test, maxTimeSteps_test, totalN_test, batchSize_test = data_batches[2]
        
        
    denseTruthIn = np.eye(4)
    hypMatSparseIn = np.eye(4)

    denseTruthIn[0,3] = 10

    truthMatSparse = denseToSparse(denseTruthIn)
    hypMatSparse = denseToSparse(hypMatSparseIn)

    test_edit_distance(truthMatSparse,hypMatSparse)

