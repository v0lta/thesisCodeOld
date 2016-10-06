import sys
#sys.path.insert(0,'../../software/' )
sys.path.append('../software/')
sys.path.append('../software/python_speech_features')
import feat
import scipy.io.wavfile as sciWav
import matplotlib.pyplot as plt
import os, glob
import kaldi_io
import numpy as np
from utils import data_lists_to_batches


def targetCodesToPhones(target,phone_map):
    code_map = {code: phone for phone, code in phone_map.items()}
    coded_targets = target_list_train[0]
    target_phones = []
    for code in coded_targets:
        target_phones.append(code_map[code])

    return target_phones

def setUpBatchLists(reader, phone_dict, phone_map, batch_count, debug=False):
    
    data = reader.scp_data
    batch_size = int(len(data)/batch_count)
    
    input_list = []
    target_list = []
    for i in range(0,batch_count*batch_size):
        #get an utterance
        utt = reader.read_next_utt()
        input_list.append(utt[1].transpose())
        
        #get the corrensponding transcription.
        transcription = phone_dict[utt[0]]
        #relace trancription strings with codes.
        target = []
        for phone in transcription:
            target.append(phone_map[phone])
        target_list.append(np.array(target, dtype = np.uint8))
 
 
    #from IPython.core.debugger import Tracer
    #Tracer()()
 
    return input_list, target_list, batch_size


def load_batched_timit39(batch_count, batch_count_val, batch_count_test, debug = False):

    fbank_Path = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/fbank"
    #open the training kaldi arks fbank files.
    train_file = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/fbank" + "/" + "raw_fbank_train.1.scp"
    train_reader = kaldi_io.KaldiReadIn(train_file)
    train_data = train_reader.scp_data
    print("Training Samples: " + str(len(train_data)))

    #open the dev fbanks
    dev_file = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/fbank" + "/" + "raw_fbank_dev.1.scp"
    dev_reader = kaldi_io.KaldiReadIn(dev_file)
    dev_data = dev_reader.scp_data
    print("Dev Samples: " + str(len(dev_data)))

    #open the test fbanks
    test_file = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/fbank" + "/" + "raw_fbank_test.1.scp"
    test_reader = kaldi_io.KaldiReadIn(test_file)
    test_data = test_reader.scp_data
    print("Test Samples: " + str(len(test_data)))

    #read the phoneme-targets.
    #for the training set.
    train_phone_path  = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/data/train/train39.text"
    train_phone_file  = open(train_phone_path)
    train_phone_lines = train_phone_file.read().splitlines()

    #for the dev set.
    dev_phone_path  = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/data/dev/dev39.text"
    dev_phone_file  = open(dev_phone_path)
    dev_phone_lines = dev_phone_file.read().splitlines()

    #for the test set.
    test_phone_path = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/data/test/test39.text"
    test_phone_file = open(test_phone_path)
    test_phone_lines = test_phone_file.read().splitlines()


    #get a dictionary connecting training utterances and transcriptions.
    train_phone_dict = {}
    for i in range(0,len(train_phone_lines)):
        tmp = train_phone_lines[i].split(' ')
        train_phone_dict.update({tmp[0]: tmp[1:]})

    #get a dictionary connecting validation utterances and transcriptions.
    dev_phone_dict = {}
    for i in range(0,len(dev_phone_lines)):
        tmp = dev_phone_lines[i].split(' ')
        dev_phone_dict.update({tmp[0]: tmp[1:]})

    #get a dictionary connecting test utterances and transcriptions.
    test_phone_dict = {}
    for i in range(0,len(test_phone_lines)):
        tmp = test_phone_lines[i].split(' ')
        test_phone_dict.update({tmp[0]: tmp[1:]})


    #check the vocabulary.
    vocabDict = {}
    for i in range(0,len(test_phone_lines)):
        tmp = test_phone_lines[i].split(' ')
        for wrd in tmp[1:]:
            if wrd in vocabDict:
                vocabDict[wrd] += 1
            else:
                if debug:
                    print('adding: ' + str({wrd: 0}))
                vocabDict.update({wrd: 0})

    if False:
        #plot the result.
        X = np.arange(len(vocabDict))
        plt.bar(X, vocabDict.values(), align='center', width=0.5)
        plt.xticks(X, vocabDict.keys())
        ymax = max(vocabDict.values()) + 1
        plt.ylim(0, ymax)
        plt.show()

    print(len(vocabDict))

    phones = list(vocabDict.keys())
    phones.sort()

    phone_map = {}
    for no,phoneme in enumerate(phones):
        phone_map.update({phoneme: no})


    maxTimeStepsTimit = 777
    #set up the training data
    input_list_train, target_list_train, batch_size = setUpBatchLists(train_reader,
                                train_phone_dict, phone_map, batch_count)

    batched_data, max_time_steps = data_lists_to_batches(input_list_train, target_list_train,
                                                    batch_size , maxTimeStepsTimit)

    training_data = (batched_data, max_time_steps, batch_size*batch_count, batch_size)

    #set up the validation data
    input_list_val, target_list_val, batch_size_val = setUpBatchLists(dev_reader, dev_phone_dict,
                                                        phone_map, batch_count_val)
    batched_data_val, max_time_steps_val = data_lists_to_batches(input_list_val, target_list_val,
                                                            batch_size_val, maxTimeStepsTimit)
    validation_data = (batched_data_val, max_time_steps_val, batch_size_val*batch_count, batch_size_val)

    #set up test data
    input_list_test, target_list_test, batch_size_test = setUpBatchLists(test_reader, test_phone_dict,
                                                        phone_map, batch_count_test)
    batched_data_test, max_time_steps_test = data_lists_to_batches(input_list_test, target_list_test,
                                                               batch_size_test, maxTimeStepsTimit)
    test_data = (batched_data_test, max_time_steps_test, batch_size_test*batch_count, batch_size_test)


    if debug == False:
        return training_data, validation_data, test_data
    else:
        target_lists = (target_list_train, target_list_val,
            target_list_test)
    return training_data, validation_data, test_data, target_lists
