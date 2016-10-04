import sys
#sys.path.insert(0,'../../software/' )
sys.path.append('../software/')
sys.path.append('../software/python_speech_features')
import feat
import os
import glob
import scipy.io.wavfile as sciWav
import matplotlib.pyplot as plt
import kaldi_io
import numpy as np
from utils import data_lists_to_batches


def set_up_batch_lists(reader, phone_dict, fold_dict, batch_count, debug=False):
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
    dev_phone_path = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/data/dev/text"
    dev_phone_file = open(dev_phone_path)
    dev_phone_lines = dev_phone_file.read().splitlines()

    #for the test set.
    test_phone_path = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/data/test/text"
    test_phone_file = open(test_phone_path)
    test_phone_lines = test_phone_file.read().splitlines()


    #get a dictionary connecting training utterances and transcriptions.
    train_phone_dict = {}
    for i in range(0, len(train_phone_lines)):
        tmp = train_phone_lines[i].split(' ')
        train_phone_dict.update({tmp[0]: tmp[1:]})

    #get a dictionary connecting validation utterances and transcriptions.
    dev_phone_dict = {}
    for i in range(0, len(dev_phone_lines)):
        tmp = dev_phone_lines[i].split(' ')
        dev_phone_dict.update({tmp[0]: tmp[1:]})

    #get a dictionary connecting test utterances and transcriptions.
    test_phone_dict = {}
    for i in range(0, len(test_phone_lines)):
        tmp = test_phone_lines[i].split(' ')
        test_phone_dict.update({tmp[0]: tmp[1:]})




    #check the vocabulary.
    vocab_dict = {}
    for i in range(0, len(test_phone_lines)):
        tmp = test_phone_lines[i].split(' ')
        for wrd in tmp[1:]:
            if wrd in vocab_dict:
                vocab_dict[wrd] += 1
            else:
                if debug:
                    print('adding: ' + str({wrd: 0}))
                vocab_dict.update({wrd: 0})

    #create a dict with alphabetically sorted keys and flod some phonemes.
    #see lee Hon page 2:
    #Among these 48 phones, there are seven groups where within-group confusions are not counted:
    # {sil, cl, vcl, epi}, {el, l}, {en, n}, {sh, zh}, {ao, aa}, {ih, ix), {ah, ax}.
    # Thus, there are effectively 39 phones that are in separate categories.
    vals = list(vocab_dict.keys())
    vals.sort()
    fold_dict = {}

    fold1 = ['sil', 'cl', 'vcl', 'epi']
    fold2 = ['el', 'l']
    fold3 = ['en', 'n']
    fold4 = ['sh', 'zh']
    fold5 = ['ao', 'aa']
    fold6 = ['ih', 'ix']
    fold7 = ['ah', 'ax']
    folds = [fold1, fold2, fold3, fold4, fold5, fold6, fold7]
    key_id = 7
    for key in vals:
        is_folded = False
        for fold_id, fold in enumerate(folds):
            if key in fold:
                fold_dict.update({key: fold_id})
                is_folded = True
        if is_folded is False:
            fold_dict.update({key: key_id})
            key_id = key_id + 1


    #if debug:
        #plot the result.
        #X = np.arange(len(vocab_dict))
        #plt.bar(X, vocab_dict.values(), align='center', width=0.5)
        #plt.xticks(X, vocab_dict.keys())
        #ymax = max(vocab_dict.values()) + 1
        #plt.ylim(0, ymax)
        #plt.show()

        #print(len(vocab_dict))


    max_time_steps_timit = 777
    #set up the training data
    input_list_train, target_list_train, batch_size =  \
        set_up_batch_lists(train_reader, train_phone_dict, fold_dict, batch_count)

    batched_data, max_time_steps = \
        data_lists_to_batches(input_list_train, target_list_train, batch_size,
                              max_time_steps_timit)

    training_data = (batched_data, max_time_steps,
                     batch_size*batch_count, batch_size)

    #set up the validation data
    input_list_val, target_list_val, batch_size_val = \
        set_up_batch_lists(dev_reader, dev_phone_dict, fold_dict, batch_count_val)
    batched_data_val, max_time_steps_val =  \
        data_lists_to_batches(input_list_val, target_list_val, batch_size_val,
                              max_time_steps_timit)
    validation_data = (batched_data_val, max_time_steps_val,
                       batch_size_val*batch_count, batch_size_val)

    #set up test data
    input_list_test, target_list_test, batch_size_test = \
        set_up_batch_lists(test_reader, test_phone_dict, fold_dict,
                           batch_count_test)
    batched_data_test, max_time_steps_test =  \
        data_lists_to_batches(input_list_test, target_list_test,
                              batch_size_test, max_time_steps_timit)
    test_data = (batched_data_test, max_time_steps_test,
                 batch_size_test*batch_count, batch_size_test)


    if debug is False:
        return training_data, validation_data, test_data
    else:
        dense_targets = (target_list_train, target_list_val,
                         target_list_test)
        return training_data, validation_data, test_data, dense_targets
