# importing libraries
import os 
#import argparser
import numpy as np
from tqdm import tqdm as prog_bar
from collections import Counter
import json
from modeling_data import *
import argparse
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Input, LSTM, RepeatVector, Reshape, concatenate
from keras.layers import RepeatVector, Dense, Activation, Input, Flatten, Reshape, Permute, Lambda, multiply
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import Bidirectional
import keras.backend as K
from utils_custom import *



if __name__=='__main__':

	PATH_TO_DATA = 'data'
	TRAIN_FILE_NAME = 'trainset.csv'
	DEV_FILE_NAME = 'devset.csv'
	TEST_FILE_NAME = 'testset.csv'
	LSTM_NUMBER_UNITS = 10
	HIDDEN_LAYER_UNITS = 10



	print('loading training data ... ')
	mapper, attributes_train, reviews_train = building_data(os.path.join(PATH_TO_DATA, TRAIN_FILE_NAME))
	print('saving mapper ...')
	# with open(os.path.join(PATH_TO_DATA, 'mapper.json'), 'w') as mapper_file : 
	# 	json.dump(mapper, fp=mapper_file)

	print('loading dev data ...')
	_, attributes_dev, reviews_dev = building_data(os.path.join(PATH_TO_DATA, TRAIN_FILE_NAME))

	print('loading test data ...')
	attributes_test = building_data_test(os.path.join(PATH_TO_DATA, TEST_FILE_NAME))

	VOCABULARY_SIZE = len(mapper)
	PADDING_LENGTH = len(max(reviews_train, key = len))
	NUMBER_OF_SAMPLES_TRAIN = len(reviews_train)
	NUMBER_OF_SAMPLES_DEV = len(reviews_dev)

	print('padding length is {}'.format(PADDING_LENGTH))
	print('vocabulary size is {}'.format(VOCABULARY_SIZE))


	print('parsing train attributes ...')
	attributes_train = parsing_list_of_attributes(attributes_train)
	print('parsing dev attributes')
	attributes_dev = parsing_list_of_attributes(attributes_dev)
	print('parsing test attributes ...')
	attributes_test = parsing_list_of_attributes(attributes_test)

	print('padding train reviews ...')
	reviews_train = padding_review(reviews_train, PADDING_LENGTH)
	print('padding dev reviews ...')
	reviews_dev = padding_review(reviews_dev, PADDING_LENGTH)

	print('mapper', mapper['<unk>'])
	
	print('mapping train reviews to index ...')
	reviews_train = np.array([np.array([mapper[token]-1 for token in review]) for review in prog_bar(reviews_train)], ndmin = 2)
	print('mapping dev reviews to index ...')
	reviews_dev = np.array([np.array([mapper[token]-1 for token in review]) for review in prog_bar(reviews_dev)], ndmin = 2)

	print('padding the train attributes')
	attributes_train = repeating_data(attributes_train)

	print('padding the dev attributes')
	attributes_dev = repeating_data(attributes_dev)

	STATE_SIZE = attributes_train.shape[2]

	input_placeholder = tf.placeholder(dtype = 'float32', shape = [None, PADDING_LENGTH, STATE_SIZE])

	output_placeholder = tf.placeholder(dtype = 'int32', shape = [None, PADDING_LENGTH])

	print(reviews_train.shape)

	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = LSTM_NUMBER_UNITS, reuse=tf.AUTO_REUSE)
	lstm_layer_output, lstm_layer_state = tf.nn.dynamic_rnn(cell = lstm_cell, 
	                                   inputs = input_placeholder, 
	                                   sequence_length = length(input_placeholder), 
	                                   dtype='float32')

	hidden_layer = tf.contrib.layers.fully_connected(inputs = lstm_layer_output, 
		num_outputs=HIDDEN_LAYER_UNITS, 
		activation_fn = tf.nn.relu)

	output_values_layer = tf.contrib.layers.fully_connected(inputs = hidden_layer, num_outputs = VOCABULARY_SIZE, activation_fn = None)
	softmax_layer = tf.nn.softmax(output_values_layer)

	indexing_layer = tf.argmax(output_values_layer, axis = 2)

	session = tf.Session()
	session.run(tf.global_variables_initializer())

	loss = tf.reduce_mean(
		tf.log(
			tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output_values_layer,
				labels = output_placeholder)))

	opt = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

	for i in range(100):
		_, loss_value, predictions = session.run([opt, loss, indexing_layer], feed_dict = {input_placeholder:attributes_train[:10], output_placeholder:reviews_train[:10]})


	"""
	# creating an input layer 
	input_layer = Input(shape=(PADDING_LENGTH, STATE_SIZE))
	# adding a padding layer 
	# ie we repeat the input vector a number PADDING_LENGTH of time

	# here comes the bidirectionnal lstm 
	bi_directionnal_layer = Bidirectional(LSTM(units = LSTM_NUMBER_UNITS, return_sequences =True))(input_layer)

	flattening_layer = Flatten()(bi_directionnal_layer)

	attention = []
	for i in range(PADDING_LENGTH):
		weighted = Dense(PADDING_LENGTH, activation='softmax')(flattening_layer)
		unfolded = Permute([2, 1])(RepeatVector(HIDDEN_LAYER_UNITS * 2)(weighted))
		multiplied = multiply([bi_directionnal_layer, unfolded])
		summed = Lambda(lambda x: K.sum(x, axis=-2))(multiplied)
		attention.append(Reshape((1, HIDDEN_LAYER_UNITS * 2))(summed))
	attention_out = concatenate(attention, axis=-2)

    # -- DECODER --
	decoder = LSTM(units=LSTM_NUMBER_UNITS,
                   dropout=0.2,
                   recurrent_dropout=0.2,
                   return_sequences=True)(attention_out)
	decoder = Dense(VOCABULARY_SIZE,
                    activation='softmax')(decoder)
	model = Model(inputs=input_layer, outputs=decoder)
	
	results = model.predict(attributes_train)
	"""

	print("##### RESULTS OF THE TRAINING ######")

	print('DONE WITH THIS SHIT')





