import sys
#sys.path.insert(0,'../../software/' )
sys.path.append('../../software/')
sys.path.append('../../software/python_speech_features')
import feat
import scipy.io.wavfile as sciWav
import matplotlib.pyplot as plt
import os, glob
import kaldi_io

fbank_Path = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/fbank"
#open the training kaldi arks fbank files.
train_file = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/fbank" + "/" + "raw_fbank_train.1.scp"
train_reader = kaldi_io.KaldiReadIn(training_file)
train_data = training_reader.scp_data
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
#for the test set.
test_phone_path = "/esat/spchtemp/scratch/moritz/dataSets/timit/s5/data/train/text"
test_phone_file = open(test_phone_path)
test_phone_lines = test_phone_file.read().splitlines()
#format this as a dict with the speaker_utterance label as key.
test_phone_dict = {}
for i in range(0,len(test_phone_lines)):
    tmp = test_phone_lines[i].split(' ')
    test_phone_dict.update({tmp[0]: tmp[1:]})


#read an utterance.
utt = train_reader.read_next_utt()
uttArray = np.asarray(utt[1]).transpose()
plt.imshow(uttArray)
plt.show()


##set up the network
#input array size is (23, 474)
