# importing libraries

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
import os 
import argparse
import numpy as np
from tqdm import tqdm as prog_bar
from collections import Counter
import json
from modeling_data import *
import argparse
import tensorflow as tf
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Input, LSTM, RepeatVector, Reshape, concatenate
from keras.layers import RepeatVector, Dense, Activation, Input, Flatten, Reshape, Permute, Lambda, multiply, dot
from keras.models import model_from_json
from keras.layers.merge import Concatenate
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import Bidirectional, TimeDistributed
import keras.backend as K
from utils_custom import *
from generation_utils import *


if __name__=='__main__':
	# defining the available modes
	available_modes = ['train', 'dev', 'test']

	parser = argparse.ArgumentParser(description='Language generation')
	parser.add_argument('--mode', help = 'choosing mode', default = 'train')
	parser.add_argument('--path_to_train', help = 'path to training data', default = 'data/trainset.csv')
	parser.add_argument('--path_to_dev', help = 'path to dev data', default = 'data/devset.csv')
	parser.add_argument('--path_to_test', help = 'path to test data', default = 'data/testset.csv')
	parser.add_argument('--pretrained', help = 'use a pretrained model [y]/[any other value]', default = 'y')
	parser.add_argument('--path_to_weights', help = 'path to the back up file for weights', default = 'model_weights.h5')
	parser.add_argument('--path_to_arch', help = 'path to the back up file for architecture', default = 'model_architecture.json')
	parser.add_argument('--path_to_output', help = 'path to output file in test mode', default = 'data/output_test.csv')
	parser.add_argument('--silent', help = 'flag for silent running [y]/[any other value]', default = 'n')
	# parsing the attributes
	args = parser.parse_args()
	MODE = args.mode
	TRAIN_FILE_NAME = args.path_to_train
	DEV_FILE_NAME = args.path_to_dev
	TEST_FILE_NAME = args.path_to_test
	PRETRAINED = args.pretrained == 'y'
	PATH_TO_ARCHITECTURE = args.path_to_arch
	PATH_TO_WEIGHTS = args.path_to_weights
	PATH_TO_OUTPUT = args.path_to_output
	silent = args.silent == 'y'
	EPOCHS = 100000
	LSTM_NUMBER_UNITS = 20

	# checking whether the mode is available
	if MODE not in available_modes:
		print('the only possible modes are :')
		print(available_modes)
		print('you have entered : "{}"\n'.format(MODE))
		exit()

	# print a summary of the choices
	print('################# SUMMARY #################')
	print('mode :\t\t\t', MODE)
	print('pretrained model :\t', PRETRAINED)
	print('path to train data : \t', TRAIN_FILE_NAME)
	print('path to dev data : \t', DEV_FILE_NAME)
	print('path to test data : \t', TEST_FILE_NAME)
	print('path to weights : \t', PATH_TO_WEIGHTS)
	print('path to architecture : \t', PATH_TO_ARCHITECTURE)
	print('nb of training epochs :\t', EPOCHS)
	print('###########################################')
	print()

	debug = False

	# loading the training data to build the mapper and get the size of the different sets
	if not silent : 
		print('loading training data ... ')
		mapper, attributes_train, reviews_train = building_data(TRAIN_FILE_NAME)
		print('parsing train attributes ...')
		attributes_train = parsing_list_of_attributes(attributes_train)
	else :
		mapper, attributes_train, reviews_train = building_data_silent(TRAIN_FILE_NAME)
		attributes_train = parsing_list_of_attributes_silent(attributes_train)

	# create a reverse mapper dictionnary to translate embedding in tokens
	reverse_mapper = {index-1:token for token, index in mapper.items()}
	# different parameters for the model
	VOCABULARY_SIZE = len(mapper)
	PADDING_LENGTH = len(max(reviews_train, key = len))
	STATE_SIZE = attributes_train.shape[-1]

	# loading or creating the model
	if not PRETRAINED:
		input_attributes = Input(shape = (PADDING_LENGTH, STATE_SIZE, ))
		lstm1 = Bidirectional(LSTM(LSTM_NUMBER_UNITS, return_sequences = True, unroll = True))(input_attributes)
		lstm2 = Bidirectional(LSTM(LSTM_NUMBER_UNITS, return_sequences = True, unroll = True))(lstm1)
		dense = Dense(LSTM_NUMBER_UNITS, activation = 'relu')(lstm2)
		dense = Dense(VOCABULARY_SIZE, activation = 'softmax')(dense)
		model = Model(input_attributes, dense)


		"""inputs = Input(shape = (PADDING_LENGTH, STATE_SIZE, ))
								per = Permute((2,1))(inputs)
								dense = Dense(PADDING_LENGTH, activation = 'softmax')(per)
								dense = Permute((2,1))(dense)
								attention = multiply([inputs, K.transpose(dense)])
								lstm = Bidirectional(LSTM(LSTM_NUMBER_UNITS, return_sequences = True))(inputs)
								lstm_with_attention = Bidirectional(LSTM(LSTM_NUMBER_UNITS, return_sequences = True))(inputs)
						
								conc = concatenate([lstm, lstm_with_attention])
						
								output = Dense(VOCABULARY_SIZE, activation = 'softmax', use_bias=False)(conc)
						
								model = Model(inputs, output)"""
		saving_model_architecture(model, PATH_TO_ARCHITECTURE)
		if not silent : print('model created')

	else : 

		with open(PATH_TO_ARCHITECTURE, 'r') as model_file:
			json_model = str(model_file.read())
		model = model_from_json(json_model)
		model.load_weights(PATH_TO_WEIGHTS)
		if not silent : print('model loaded')

	# compiling the model
	model.compile(optimizer = 'rmsprop', loss = ['categorical_crossentropy'])
	if not silent : model.summary()


	# loading data
	if MODE == 'train':
		attributes = attributes_train
		reviews = reviews_train
	elif MODE == 'dev':
		if not silent : 
			print('loading dev data ...')
			_, attributes, reviews = building_data(DEV_FILE_NAME)
			attributes = parsing_list_of_attributes(attributes)
		else :
			_, attributes, reviews = building_data_silent(DEV_FILE_NAME)
			attributes = parsing_list_of_attributes_silent(attributes)
	elif MODE == 'test' :
		if not silent :
			attributes = building_data_test(TEST_FILE_NAME)
			attributes, attributes_dictionnary = parsing_list_of_attributes_test(attributes)

			reviews = False
		else :
			attributes = building_data_test_silent(TEST_FILE_NAME)
			attributes, attributes_dictionnary = parsing_list_of_attributes_test_silent(attributes)

			reviews = False

	# padding and mapping the reviews
	if reviews : 
		if not silent : 
			print('padding reviews ...')
			reviews = padding_review(reviews, PADDING_LENGTH)
			print('mapping reviews to index ...')
			reviews = mapping_reviews(list_of_reviews=reviews, mapper=mapper, review_len = PADDING_LENGTH)

		else :
			reviews = padding_review_silent(reviews, PADDING_LENGTH)
			reviews = mapping_reviews_silent(list_of_reviews=reviews, mapper=mapper, review_len = PADDING_LENGTH)
	# print(attributes.shape)
	# shaping the input so it has the same size as the output
	if not silent : print('padding attributes ...')
	attributes = repeating_data(attributes)


	# training mode 
	if MODE == 'train':
		for index_epoch in range(1, EPOCHS+1):
			list_of_batch_index = generate_batches_index(number_of_samples = VOCABULARY_SIZE, batch_size = 10)
			for index_batch, batch_indexes in enumerate(list_of_batch_index):
				if debug : batch_indexes = [i for i in range(10)]
				model.fit(attributes[batch_indexes,:], to_categorical(reviews[batch_indexes,:], num_classes = VOCABULARY_SIZE), 
					epochs = 10, verbose = False)
				if (index_batch % 10 == 0) and (not silent):


					predictions = model.predict(attributes[batch_indexes[:1],:,:])
					predictions = np.argmax(predictions, axis = -1)
					real_review = reviews[batch_indexes[:1],:]
					tokens = generate_list_of_tokens(predictions, reverse_mapper)
					review_pred = create_review(tokens[0])
					review_act = generate_list_of_tokens(real_review, reverse_mapper)
					review_act = create_review(review_act[0])

					print('epoch {}/{} - batch {}/{}'.format(index_epoch, EPOCHS, index_batch, len(list_of_batch_index)))
					print('actual : ' + review_act)
					print('predicted : ' + review_pred)
			
				if index_batch %100==0: 
					model.save_weights(PATH_TO_WEIGHTS)
					if not silent : print('model weights saved')
	elif MODE == 'dev':
		NB_OF_SAMPLES = attributes.shape[0]
		predictions = []
		for i in range(NB_OF_SAMPLES//100):
			predictions += [model.predict(attributes[(i*100):((i+1)*100),:,:])]

			if not silent:
				predictions_tokens = predictions[-1]
				predictions_tokens = np.argmax(predictions_tokens, axis = -1)
				real_review = reviews[(i+1)*100-1:(i+1)*100,:]
				tokens = generate_list_of_tokens(predictions_tokens, reverse_mapper)
				review_pred = create_review(tokens[0])
				review_act = generate_list_of_tokens(real_review, reverse_mapper)
				review_act = create_review(review_act[0])
				print('actual : ' + review_act)
				print('predicted : ' + review_pred)


		predictions += [model.predict(attributes[(i+1)*100:(i+2)*100,:,:])]
		rouge_scores_1 = []
		rouge_scores_2 = []
		rouge_scores_3 = []
		bleu_score = []
		for index_batch, predictions_batch in prog_bar(enumerate(predictions)):
			for index, predicted_review in enumerate(predictions_batch):
				predicted_review_tokens = np.argmax(predicted_review, axis = -1)
				actual_review = reviews[index_batch*100+index]
				rouge_scores_1.append(compute_rouge_score(actual_review, predicted_review_tokens, n = 1))
				rouge_scores_2.append(compute_rouge_score(actual_review, predicted_review_tokens, n = 2))
				rouge_scores_3.append(compute_rouge_score(actual_review, predicted_review_tokens, n = 3))
				bleu_score.append(compute_bleu_score(actual_review, predicted_review_tokens))

		print('rouge score 1 :\t', np.mean(rouge_scores_1))
		print('rouge score 2 :\t', np.mean(rouge_scores_2))
		print('rouge score 3 :\t', np.mean(rouge_scores_3))
		print('bleu score :\t', np.mean(bleu_score))

	elif MODE == 'test':
		test_predictions = model.predict(attributes)
		test_predictions = np.argmax(test_predictions, axis = -1)
		print(test_predictions[0])
		test_predictions = generate_list_of_tokens(test_predictions, reverse_mapper)
		# test_predictions = [create_review(pred,*attributes_dictionnary[index]) for index, pred in enumerate(test_predictions)]
		
		with open(PATH_TO_OUTPUT, 'w') as output_results : 
			for index, test_tokens in enumerate(test_predictions):
				dictionnary = attributes_dictionnary[index]
				
				review = create_review(test_tokens, name = dictionnary['name'], area = dictionnary['area'], near = dictionnary['near'])
				line = str(dictionnary) +'\t'+ review + '\n'
				output_results.write(line)
				if not silent : print(line, end = '')
			
		print('cool')



	print("© Paul Déchorgnat et al.")






