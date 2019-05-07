import operator
import random
import time
import csv
import sys

import numpy as np
import tensorflow as tf
import librosa

#from audio_reader import AudioReader
#from file_logger import FileLogger
from utils import FIRST_INDEX, sparse_tuple_from
#from utils import convert_inputs_to_ctc_format
from python_speech_features import mfcc
from WordBeamSearch import wordBeamSearch
from LanguageModel import LanguageModel
import codecs




sample_rate = 16000
# Some configs
num_features = 26  # log filter bank or MFCC features
# Accounting the 0th index +  space + blank label = 28 characters
num_classes = ord('z') - ord('a') + 1 + 1 + 1

# Hyper-parameters
num_epochs = 100000
num_hidden = 256
batch_size = 16

num_examples = 1
num_batches_per_epoch = 10

# make sure the values match the ones in generate_audio_cache.py
"""audio = AudioReader(audio_dir=None,
                    cache_dir='cache',
                    sample_rate=sample_rate)"""

#file_logger = FileLogger('out.tsv', ['curr_epoch', 'train_cost', 'train_ler', 'val_cost', 'val_ler'])

def softmax(mat):
	"calc softmax such that labels per time-step form probability distribution"
	# dim0=t, dim1=c
	maxT,_=mat.shape
	res=np.zeros(mat.shape)
	for t in range(maxT):
		y=mat[t,:]
		maxValue = np.max(y)
		e=np.exp(y - maxValue)
		s=np.sum(e)
		res[t,:]=e/s

	return res

def next_batch(bs=batch_size, train=True):
    x_batch = []
    y_batch = []
    seq_len_batch = []
    original_batch = []
    for k in range(bs):
        ut_length_dict = dict([(k, len(v['target'])) for (k, v) in audio.cache.items()])
        utterances = sorted(ut_length_dict.items(), key=operator.itemgetter(1))
        test_index = 15
        if train:
            utterances = [a[0] for a in utterances[test_index:]]
        else:
            utterances = [a[0] for a in utterances[:test_index]]
        random_utterance = random.choice(utterances)
        training_element = audio.cache[random_utterance]
        target_text = training_element['target']
        if train:
            l_shift = np.random.randint(low=1, high=1000)
            audio_buffer = training_element['audio'][l_shift:]
        else:
            audio_buffer = training_element['audio']
        x, y, seq_len, original = convert_inputs_to_ctc_format(audio_buffer,
                                                               sample_rate,
                                                               target_text,
                                                               num_features)
        x_batch.append(x)
        y_batch.append(y)
        seq_len_batch.append(seq_len)
        original_batch.append(original)

    # Creating sparse representation to feed the placeholder
    # inputs = np.concatenate(x_batch, axis=0)
    y_batch = sparse_tuple_from(y_batch)
    seq_len_batch = np.array(seq_len_batch)[:, 0]
    for i, pad in enumerate(np.max(seq_len_batch) - seq_len_batch):
        x_batch[i] = np.pad(x_batch[i], ((0, 0), (0, pad), (0, 0)), mode='constant', constant_values=0)

    x_batch = np.concatenate(x_batch, axis=0)
    # return np.array(list(x_batch[0]) * batch_size), y_batch, np.array(seq_len_batch[0] * batch_size), original_batch
    # np.pad(x_batch[0], ((0, 0), (10, 0), (0, 0)), mode='constant', constant_values=0)

    return x_batch, y_batch, seq_len_batch, original_batch


def decode_batch(d, original, phase='training'):
    aligned_original_string = ''
    aligned_decoded_string = ''
    for jj in range(batch_size)[0:2]:  # just for visualisation purposes. we display only 2.
        values = d.values[np.where(d.indices[:, 0] == jj)[0]]
        str_decoded = ''.join([chr(x) for x in np.asarray(values) + FIRST_INDEX])
        # Replacing blank label to none
        str_decoded = str_decoded.replace(chr(ord('z') + 1), '')
        # Replacing space label to space
        str_decoded = str_decoded.replace(chr(ord('a') - 1), ' ')
        maxlen = max(len(original[jj]), len(str_decoded))
        aligned_original_string += str(original[jj]).ljust(maxlen) + ' | '
        aligned_decoded_string += str(str_decoded).ljust(maxlen) + ' | '
    print('- Original (%s) : %s ...' % (phase, aligned_original_string))
    print('- Decoded  (%s) : %s ...' % (phase, aligned_decoded_string))


def run_ctc():
    file_wav = sys.argv[1]
    audio, _ = librosa.load(file_wav, sr=sample_rate, mono=True)
    audio = audio.reshape(-1, 1)
    inputs = mfcc(audio, samplerate=sample_rate, numcep=num_features)
    train_inputs = np.asarray(inputs[np.newaxis, :])
    train_inputs = (train_inputs - np.mean(train_inputs)) / np.std(train_inputs)
    train_seq_len = [train_inputs.shape[1]]
    x_batch = []
    seq_len_batch = []
    x_batch.append(train_inputs)
    seq_len_batch.append(train_seq_len)
    seq_len_batch = np.array(seq_len_batch)[:, 0]
    for i, pad in enumerate(np.max(seq_len_batch) - seq_len_batch):
        x_batch[i] = np.pad(x_batch[i], ((0, 0), (0, pad), (0, 0)), mode='constant', constant_values=0)
    x_batch = np.concatenate(x_batch, axis=0)



    graph = tf.Graph()
    with graph.as_default():
        # e.g: log filter bank or MFCC features
        # Has size [batch_size, max_step_size, num_features], but the
        # batch_size and max_step_size can vary along each step
        inputs = tf.placeholder(tf.float32, [None, None, num_features], name='inputs')

        # Here we use sparse_placeholder that will generate a
        # SparseTensor required by ctc_loss op.
        # https://www.tensorflow.org/api_docs/python/tf/sparse/SparseTensor
        # https://www.tensorflow.org/api_docs/python/tf/nn/ctc_loss
        #targets = tf.sparse_placeholder(tf.int32, name='targets')

        # 1d array of size [batch_size]
        seq_len = tf.placeholder(tf.int32, [None], name='seq_len')

        # Defining the cell
        # Can be:
        #   tf.nn.rnn_cell.RNNCell
        #   tf.nn.rnn_cell.GRUCell
        cell = tf.contrib.rnn.LSTMCell(num_hidden, state_is_tuple=True)

        # Stacking rnn cells
        stack = tf.contrib.rnn.MultiRNNCell([cell], state_is_tuple=True)

        # The second output is the last state and we will no use that
        outputs, _ = tf.nn.dynamic_rnn(stack, inputs, seq_len, dtype=tf.float32)

        shape = tf.shape(inputs)
        batch_s, max_time_steps = shape[0], shape[1]

        # Reshaping to apply the same weights over the timesteps
        outputs = tf.reshape(outputs, [-1, num_hidden])

        # Truncated normal with mean 0 and stdev=0.1
        # Tip: Try another initialization
        # see https://www.tensorflow.org/versions/r0.9/api_docs/python/contrib.layers.html#initializers
        W = tf.Variable(tf.truncated_normal([num_hidden,
                                             num_classes],
                                            stddev=0.1))
        # Zero initialization
        # Tip: Is tf.zeros_initializer the same?
        b = tf.Variable(tf.constant(0., shape=[num_classes]))

        # Doing the affine projection
        logits = tf.matmul(outputs, W) + b

        #logits_ori = logits

        # Reshaping back to the original shape
        #logits = tf.reshape(logits, [batch_s, -1, num_classes])

        # Time major
        #logits = tf.transpose(logits, (1, 0, 2))

        #loss = tf.nn.ctc_loss(targets, logits, seq_len)
        #cost = tf.reduce_mean(loss)

        # optimizer = tf.train.AdamOptimizer().minimize(cost)
        # optimizer = tf.train.MomentumOptimizer(learning_rate=0.01, momentum=0.9).minimize(cost)
        #optimizer = tf.train.AdamOptimizer(learning_rate=5e-4).minimize(cost)

        # Option 2: tf.contrib.ctc.ctc_beam_search_decoder
        # (it's slower but you'll get better results)
        #decoded, log_prob = tf.nn.ctc_greedy_decoder(logits, seq_len)

        # Inaccuracy: label error rate
        #ler = tf.reduce_mean(tf.edit_distance(tf.cast(decoded[0], tf.int32),
        #                                      targets))
        saver = tf.train.Saver()

    with tf.Session(graph=graph) as session:

        tf.global_variables_initializer().run()
        saver.restore(session,"model/model.ckpt")


        #val_inputs, val_targets, val_seq_len, val_original = next_batch(train=False)
        val_feed = {inputs: x_batch,
                    seq_len: seq_len_batch}

        #val_cost, val_ler = session.run([cost, ler], feed_dict=val_feed)

        # Decoding
        # np.where(np.diff(d.indices[:, 0]) == 1)
        lo_ori = session.run(logits, feed_dict=val_feed)
        #decode_batch(d, val_original, phase='validation')
        count = 0
        print (lo_ori)
        file = open('decoded_data.csv', mode='w')
        writer = csv.writer(file, delimiter=';', quotechar='"', quoting=csv.QUOTE_MINIMAL)

        for lo in lo_ori:
            count = count + 1
            writer.writerow(lo)
        print (count)
        mat=softmax(lo_ori)
        chars_ = ' abcdefghijklmnopqrstuvwxyz'
        wordChars = codecs.open('data/bentham/wordChars.txt', 'r', 'utf8').read()
        useNGrams = True
        lm = LanguageModel(codecs.open('data/bentham/corpus.txt', 'r', 'utf8').read(), chars_, wordChars)
        res=wordBeamSearch(mat, 10, lm, useNGrams)
        print('Result:       "'+res+'"')



        """print('-' * 80)
        log = "Epoch {}/{}, train_cost = {:.3f}, train_ler = {:.3f}, " \
              "val_cost = {:.3f}, val_ler = {:.3f}, time = {:.3f}"

        file_logger.write([curr_epoch + 1,
                           train_cost,
                           train_ler,
                           val_cost,
                           val_ler])

        print(log.format(curr_epoch + 1, num_epochs, train_cost, train_ler,
                         val_cost, val_ler, time.time() - start))"""


if __name__ == '__main__':
    run_ctc()
