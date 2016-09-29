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


def setUpBatchLists(reader, phone_dict, fold_dict, batch_count, debug=False):
    '''
    Function which generates batch data using an ark reader
    and a phone-map dictionary.
    @param reader: an ark reader instance
    @param phone_dict: a phoneme dictionary with a key for
                       every phoneme in the target data.
    '''

    data = reader.scp_data
    batch_size = int(len(data)/batch_count)

    input_list = []
    target_list = []
    for _ in range(0, batch_count*batch_size):
        #get an utterance
        utt = reader.read_next_utt()
        input_list.append(utt[1].transpose())

        #get the corrensponding transcription.
        transcription = phone_dict[utt[0]]
        #relace trancription strings with codes.
        target = []
        for phone in transcription:
            target.append(fold_dict[phone])
        target_list.append(np.array(target, dtype=np.uint8))


    return input_list, target_list, batch_size


def load_batched_timit(batch_count, batch_count_val, batch_count_test, debug=False):

    fbank_Path = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/fbank"
    #open the training kaldi arks fbank files.
    train_file = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/fbank" \
                  + "/" + "raw_fbank_train.1.scp"
    train_reader = kaldi_io.KaldiReadIn(train_file)
    train_data = train_reader.scp_data
    print("Training Samples: " + str(len(train_data)))

    #open the dev fbanks
    dev_file = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/fbank" \
                + "/" + "raw_fbank_dev.1.scp"
    dev_reader = kaldi_io.KaldiReadIn(dev_file)
    dev_data = dev_reader.scp_data
    print("Dev Samples: " + str(len(dev_data)))

    #open the test fbanks
    test_file = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/fbank"  \
                + "/" + "raw_fbank_test.1.scp"
    test_reader = kaldi_io.KaldiReadIn(test_file)
    test_data = test_reader.scp_data
    print("Test Samples: " + str(len(test_data)))

    #read the phoneme-targets.
    #for the training set.
    train_phone_path = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/data/train/text"
    train_phone_file = open(train_phone_path)
    train_phone_lines = train_phone_file.read().splitlines()

    #for the dev set.
    dev_phone_path  = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/data/dev/text"
    dev_phone_file  = open(dev_phone_path)
    dev_phone_lines = dev_phone_file.read().splitlines()

    #for the test set.
    test_phone_path = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/data/test/text"
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

    #create a dict with alphabetically sorted keys and flod some phonemes.
    #see lee Hon page 2:
    #Among these 48 phones, there are seven groups where within-group confusions are not counted:
    # {sil, cl, vcl, epi}, {el, l}, {en, n}, {sh, zh}, {ao, aa}, {ih, ix), {ah, ax}.
    # Thus, there are effectively 39 phones that are in separate categories.
    vals = list(vocabDict.keys())
    vals.sort()
    fold_dict = {}

    fold1 = ['sil','cl','vcl','epi']
    fold2 = ['el','l']
    fold3 = ['en','n']
    fold4 = ['sh','zh']
    fold5 = ['ao','aa']
    fold6 = ['ih','ix']
    fold7 = ['ah','ax']
    folds = [fold1, fold2, fold3, fold4, fold5, fold6, fold7]
    keyId = 7
    for key in vals:
        foldedKey = False
        for foldId,fold in enumerate(folds):
            if key in fold:
                fold_dict.update({key: foldId})
                foldedKey = True
        if foldedKey == False:
            fold_dict.update({key: keyId})
            keyId = keyId + 1


    #if debug:
        #plot the result.
        #X = np.arange(len(vocabDict))
        #plt.bar(X, vocabDict.values(), align='center', width=0.5)
        #plt.xticks(X, vocabDict.keys())
        #ymax = max(vocabDict.values()) + 1
        #plt.ylim(0, ymax)
        #plt.show()

        #print(len(vocabDict))


    maxTimeStepsTimit = 777
    #set up the training data
    input_list_train, target_list_train, batch_size = setUpBatchLists(train_reader, train_phone_dict,
                                                            fold_dict, batch_count)

    batched_data, max_time_steps = data_lists_to_batches(input_list_train, target_list_train,
                                                        batch_size , maxTimeStepsTimit)
        
    training_data = (batched_data, max_time_steps, batch_size*batch_count, batch_size)
    
    #set up the validation data
    input_list_val, target_list_val, batch_size_val = setUpBatchLists(dev_reader, dev_phone_dict,
                                                            fold_dict, batch_count_val)
    batched_data_val, max_time_steps_val = data_lists_to_batches(input_list_val, target_list_val,
                                                                batch_size_val, maxTimeStepsTimit)
    validation_data = (batched_data_val, max_time_steps_val, batch_size_val*batch_count, batch_size_val)

    #set up test data
    input_list_test, target_list_test, batch_size_test = setUpBatchLists(test_reader, test_phone_dict,
                                                            fold_dict, batch_count_test)
    batched_data_test, max_time_steps_test = data_lists_to_batches(input_list_test, target_list_test,
                                                                   batch_size_test, maxTimeStepsTimit)
    test_data = (batched_data_test, max_time_steps_test, batch_size_test*batch_count, batch_size_test)


    if debug == False:
        return training_data, validation_data, test_data
    else:
        dense_targets = (target_list_train, target_list_val,
                target_list_test)
        return training_data, validation_data, test_data, dense_targets
