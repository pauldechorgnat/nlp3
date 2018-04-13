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
from keras.utils.np_utils import to_categorical
from utils_custom import *




if __name__=='__main__':

	parser = argparse.ArgumentParser(description='data formation')
	parser.add_argument('--format_data', metavar='dr', type=str, default='y')
	parser.add_argument('--datafolder', metavar='path_to_folder', type=str, default='data')
	parser.add_argument('--datafile', metavar="path_to_file", type = str, default = 'trainset.csv')
	parser.add_argument('--build_mapper', metavar= "building_mapper", type = str, default = 'y')

	args = parser.parse_args()
	PATH_TO_DATA_FOLDER = args.datafolder
	PATH_TO_DATA_FILE = os.path.join(PATH_TO_DATA_FOLDER, args.datafile)

	if args.format_data == "y":
		mapper, list_of_attributes, list_of_reviews = building_data(PATH_TO_DATA_FILE)
		# saving the mapper
		if args.build_mapper == 'y':
			with open(os.path.join(PATH_TO_DATA_FOLDER, 'mapper.json'), 'w') as json_file:
				json.dump(mapper, fp = json_file)
			print('mapper saved')
		VOCABULARY_SIZE = len(mapper)
		PADDING_LENGTH = len(max(list_of_reviews, key = len))
		NUMBER_OF_SAMPLES = len(list_of_reviews)

		# padding the reviews
		list_of_reviews = [padding_review(review, PADDING_LENGTH) for review in list_of_reviews]
		# mapping the reviews to integers
		list_of_reviews = [[mapper[token]-1 for token in review] for review in list_of_reviews]
		# saving the reviews 
		print('saving transformed reviews ...')
		with open(os.path.join(PATH_TO_DATA_FOLDER, 'transformed_reviews'), 'w') as file : 
			for review in prog_bar(list_of_reviews):
				file.write(' '.join([str(token) for token in review])+ '\n')
		print('transformed reviews saved')

		# parsing attributes 
		print('parsing attributes')
		list_of_attributes = np.array([parsing_attributes(attribute) for attribute in prog_bar(list_of_attributes)]).reshape(NUMBER_OF_SAMPLES, -1)
		print(list_of_attributes.shape)
		with open(os.path.join(PATH_TO_DATA_FOLDER, 'attributes'), 'w') as file: 
			np.savetxt(fname = file, X = list_of_attributes, fmt = '%u')
		print('attributes saved')

	else : 
		
		list_of_reviews = []
		with open(os.path.join(PATH_TO_DATA_FOLDER, 'transformed_reviews'), 'r') as review_file:
			for line in prog_bar(review_file):
				list_of_reviews += [[int(val) for val in str(line).split(' ')]]
		print('reviews loaded')
		list_of_attributes = []
		with open(os.path.join(PATH_TO_DATA_FOLDER, 'attributes'), 'r') as attribute_file:
			for line in prog_bar(attribute_file):
				list_of_attributes+=[[int(val) for val in str(line).split(' ')]]
		list_of_attributes = np.array(list_of_attributes)
		print('attributes loaded')

	# loading the mapper
	with open(os.path.join(PATH_TO_DATA_FOLDER, 'mapper.json'), 'r') as json_file:
			mapper = dict(json.load(json_file))
	print('mapper loaded')

	VOCABULARY_SIZE = len(mapper)
	PADDING_LENGTH = len(max(list_of_reviews, key = len))
	NUMBER_OF_SAMPLES = len(list_of_reviews)
	print('padding length : {}'.format(PADDING_LENGTH))
	print('vocabulary size : {}'.format(VOCABULARY_SIZE))
	print('number of reviews : {}'.format(NUMBER_OF_SAMPLES))


	print("\n"*5)
	print("#"*30)
	print('DATA MODELING DONE')
	print("#"*30)


	STATE_SIZE = 21
	LSTM_UNITS = 100
	HIDDEN_UNITS = 100
	


	session = tf.Session()
	# input placeholder
	input_layer = tf.placeholder(dtype = 'float32', shape = [None,1, STATE_SIZE], name = 'input_layer')
	# we need to repeat the input to get the same number of input as the output
	# so we add a tiling layer
	tiling_layer = tf.tile(input=input_layer, multiples = [1, PADDING_LENGTH, 1])
	# creating LSTM cells 
	cell = tf.nn.rnn_cell.BasicLSTMCell(num_units = LSTM_UNITS, reuse=tf.AUTO_REUSE)#,
		#initializer=tf.constant_initializer(value=0, dtype=tf.int32))
	# creating LSTM - RNN layer
	lstm_layer_output, lstm_layer_state = tf.nn.dynamic_rnn(cell = cell, 
	                                   inputs = tiling_layer, 
	                                   sequence_length = length(tiling_layer), 
	                                   dtype='float32')

	hidden_layer = tf.contrib.layers.fully_connected(inputs = lstm_layer_output, num_outputs=HIDDEN_UNITS, activation_fn = tf.nn.relu)
	output_layer = tf.contrib.layers.fully_connected(inputs = hidden_layer, num_outputs= VOCABULARY_SIZE, activation_fn = tf.nn.relu)

	output_values = tf.placeholder(dtype = 'int64', shape = [None, PADDING_LENGTH], name = 'output_values')

	loss = tf.reduce_mean(tf.log(tf.nn.sparse_softmax_cross_entropy_with_logits(logits = output_layer+tf.constant(value = .000001), 
	                                                                          	labels = output_values)))
	opt = tf.train.AdamOptimizer(learning_rate = 0.001).minimize(loss)

	session.run(tf.global_variables_initializer())

	list_of_reviews = np.array(list_of_reviews).reshape(-1,PADDING_LENGTH)
	list_of_attributes = list_of_attributes.reshape(-1, 1, STATE_SIZE)
	number_of_epochs = 100
	print(list_of_attributes.shape)
	print(list_of_attributes[:3])
	print(list_of_reviews.shape)
	print(list_of_reviews[:3])
	for i in range(1000):
		results = session.run([tiling_layer, loss, opt, output_values], 
			feed_dict = {input_layer:list_of_attributes[:,:,:], output_values : list_of_reviews[:,:]})
		print(results[1])