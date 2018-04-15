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
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import Bidirectional
import keras.backend as K
from utils_custom import *
from generation_utils import *



if __name__=='__main__':

	PATH_TO_DATA = 'data'
	TRAIN_FILE_NAME = 'trainset.csv'
	DEV_FILE_NAME = 'devset.csv'
	TEST_FILE_NAME = 'testset.csv'
	LSTM_NUMBER_UNITS = 100
	HIDDEN_LAYER_UNITS = 100



	print('loading training data ... ')
	mapper, attributes_train, reviews_train = building_data(os.path.join(PATH_TO_DATA, TRAIN_FILE_NAME))
	# print('saving mapper ...')
	# with open(os.path.join(PATH_TO_DATA, 'mapper.json'), 'w') as mapper_file : 
	# 	json.dump(mapper, fp=mapper_file)
	reverse_mapper = {index-1:token for token, index in mapper.items()}
	'''print('loading dev data ...')
	_, attributes_dev, reviews_dev = building_data(os.path.join(PATH_TO_DATA, TRAIN_FILE_NAME))

	print('loading test data ...')
	attributes_test = building_data_test(os.path.join(PATH_TO_DATA, TEST_FILE_NAME))
	'''
	VOCABULARY_SIZE = len(mapper)
	PADDING_LENGTH = len(max(reviews_train, key = len))
	NUMBER_OF_SAMPLES_TRAIN = len(reviews_train)
	'''NUMBER_OF_SAMPLES_DEV = len(reviews_dev)'''

	print('## padding length : {}'.format(PADDING_LENGTH))
	print('## vocabulary size : {}'.format(VOCABULARY_SIZE))


	print('parsing train attributes ...')
	attributes_train = parsing_list_of_attributes(attributes_train)
	'''print('parsing dev attributes')
	attributes_dev = parsing_list_of_attributes(attributes_dev)
	print('parsing test attributes ...')
	attributes_test = parsing_list_of_attributes(attributes_test)'''

	print('padding train reviews ...')
	reviews_train = padding_review(reviews_train, PADDING_LENGTH)
	'''print('padding dev reviews ...')
	reviews_dev = padding_review(reviews_dev, PADDING_LENGTH)'''

	print('mapping train reviews to index ...')
	reviews_train = mapping_reviews(list_of_reviews=reviews_train, mapper=mapper, review_len = PADDING_LENGTH)
	'''print('mapping dev reviews to index ...')
	reviews_dev = mapping_reviews(list_of_reviews=reviews_dev, mapper=mapper, review_len = PADDING_LENGTH)'''

	print('padding the train attributes')
	attributes_train = repeating_data(attributes_train)
	'''print('padding the dev attributes')
	attributes_dev = repeating_data(attributes_dev)'''

	STATE_SIZE = attributes_train.shape[2]
	print('## state size : {}'.format(STATE_SIZE))




	inputs = Input(shape = (PADDING_LENGTH, STATE_SIZE, ))
	per = Permute((2,1))(inputs)
	# per = Reshape((STATE_SIZE, PADDING_LENGTH))(per)
	dense = Dense(PADDING_LENGTH, activation = 'softmax')(per)
	dense = Permute((2,1))(dense)
	attention = multiply([inputs, K.transpose(dense)])

	lstm = Bidirectional(LSTM(LSTM_NUMBER_UNITS, return_sequences = True))(inputs)
	lstm_with_attention = Bidirectional(LSTM(LSTM_NUMBER_UNITS, return_sequences = True))(inputs)

	conc = concatenate([lstm, lstm_with_attention])
	output_layers=[]
	# for i in range(PADDING_LENGTH):
	# 	output_layers.append(Dense(VOCABULARY_SIZE, activation = 'softmax', use_bias=False)(conc))

	#output = Concatenate()(output_layers)
	output = Dense(VOCABULARY_SIZE, activation = 'softmax', use_bias=False)(conc)

	model = Model(inputs, output)
	model.compile(optimizer = 'rmsprop', loss = ['categorical_crossentropy'])
	model.summary()
	print(reviews_train.shape)
	for i in range(2000):
		model.fit(attributes_train[:100,:,:], to_categorical(reviews_train[:100,:], num_classes = VOCABULARY_SIZE), epochs = 10, verbose = False)
		predictions = model.predict(attributes_train[:1,:,:])
		predictions = np.argmax(predictions, axis = -1)
		tokens = generate_list_of_tokens(predictions, reverse_mapper)
		review_pred = create_review(tokens[0])
		print(i, 'results :', review_pred)

	'''
	session = tf.Session()
	# input placeholder
	input_placeholder = tf.placeholder(dtype = 'float64', shape = [None, PADDING_LENGTH, STATE_SIZE])
	# output placeholder
	output_placeholder = tf.placeholder(dtype = 'int32', shape = [None, PADDING_LENGTH])
	# defining lstm cells
	lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = LSTM_NUMBER_UNITS, reuse=tf.AUTO_REUSE)
	
	# ENCODER 
	outputs_enc_rnn, states_enc_rnn = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_cell,
		cell_bw = lstm_cell,
		inputs = input_placeholder, 
		sequence_length = length(input_placeholder), 
		dtype='float64')
	enc_layer = tf.concat(outputs_enc_rnn, 2)
	# DECODER 
	outputs_dec_rnn, states_dec_rnn = tf.nn.bidirectional_dynamic_rnn(cell_fw = lstm_cell,
		cell_bw = lstm_cell,
		inputs = input_placeholder, 
		sequence_length = length(input_placeholder), 
		dtype='float64')
	dec_layer = tf.concat(outputs_dec_rnn, 2)
	
	Wc = tf.contrib.layers.fully_connected(inputs = enc_layer, 
		activation_fn = None, 
		biases_initializer = None, 
		num_outputs = 2*LSTM_NUMBER_UNITS)
	Wh = tf.contrib.layers.fully_connected(inputs = dec_layer, 
		activation_fn = None, 
		biases_initializer = None,  
		num_outputs = 2*LSTM_NUMBER_UNITS)
	hidden_layer = tf.contrib.layers.bias_add(inputs = Wc+Wh, activation_fn = tf.tanh)
	
	# hidden_layer = tf.contrib.layers.fully_connected(inputs = concatenating_layer, 
	# 	num_outputs=HIDDEN_LAYER_UNITS, 
	# 	activation_fn = tf.nn.relu)
	session.run(tf.global_variables_initializer())
	results = session.run([hidden_layer], feed_dict = {input_placeholder:attributes_train[:10,:,:], 
				output_placeholder:reviews_train[:10,:]})
	print(results[0].shape)

	raise ShogunError
	output_values_layer = tf.contrib.layers.fully_connected(inputs = hidden_layer, num_outputs = VOCABULARY_SIZE)
	

	indexing_layer = tf.argmax(output_values_layer, axis = 2)

	loss = tf.reduce_mean(
		tf.log(
			tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output_values_layer,
				labels = output_placeholder)))
	# RMSProp
	opt = tf.train.RMSPropOptimizer(learning_rate = 0.00001).minimize(loss)
	session.run(tf.global_variables_initializer())

	NUMBER_OF_EPOCHS = 2

	for epoch in range(NUMBER_OF_EPOCHS):
		batch_indexes = generate_batches_index(number_of_samples = NUMBER_OF_SAMPLES_TRAIN, batch_size = 20)
		nb_of_batches = len(batch_indexes)
		print(batch_indexes)
		for j, index in enumerate(batch_indexes):

			_, loss_value, predictions = session.run([opt, loss, indexing_layer], 
				feed_dict = {input_placeholder:attributes_train[index,:,:], 
				output_placeholder:reviews_train[index,:]})

			print('epoch {}/{} - batch {}/{} : {}'.format(epoch+1, NUMBER_OF_EPOCHS, j+1, nb_of_batches, loss_value))
			print(predictions.shape)
			print(create_review(generate_list_of_tokens(predictions, reverse_mapper)[0]))
	


	'''
	print("##### RESULTS OF THE TRAINING ######")

	print('DONE WITH THIS SHIT')





