# importing libraries

import warnings
warnings.simplefilter(action='ignore', category=FutureWarning)
warnings.simplefilter(action='ignore', category=UserWarning)
import os 
import argparse
import numpy as np
from tqdm import tqdm as prog_bar
import json
import argparse
from keras.preprocessing.sequence import pad_sequences
from keras.layers import Dense, Dropout, Input, LSTM, Flatten, Embedding
from keras.models import model_from_json
from keras.models import Model
from keras.utils.np_utils import to_categorical
from keras.layers.wrappers import Bidirectional
import keras.backend as K
from keras.optimizers import RMSprop

from utils_new import *

if __name__=='__main__':

	parser = argparse.ArgumentParser(description='Language generation')
	parser.add_argument("--train_dataset", help="Path to the training set", default = 'data/trainset.csv', type = str)
	parser.add_argument("--path_to_mapper", help="Path to the mapper", default = 'mapper.json', type = str)
	parser.add_argument('--path_to_model', type = str, default = 'model', help='Path to the model')
	parser.add_argument('--nepochs', type=int, metavar='n', default='100', help='number of epochs')
	parser.add_argument('--lstm_units', type = int, default = '100', help='number of lstm units')
	parser.add_argument('--silent', type = bool, default = False, help='silent or not')
	parser.add_argument('--batch_size', type = int, default = '10', help='batch size')
	parser.add_argument("--load_model", type=bool,  help="load a pretrained model to carry training on it", default = False)
	


	args = parser.parse_args()
	
	PATH_TO_DATA = args.train_dataset #'data/trainset.csv'
	PATH_TO_MAPPER = args.path_to_mapper #'mapper.json'
	PATH_TO_MODEL = args.path_to_model # 'model'
	LSTM_NUMBER_UNITS = args.lstm_units #100
	BATCH_SIZE = args.batch_size # 10
	EPOCHS = args.nepochs # 100
	SILENT = args.silent
	LOAD_MODEL = args.load_model # False

	# creating a folder to save the model 
	try : 
		os.stat(PATH_TO_MODEL)
	except :
		print('creating "{}" folder'.format(PATH_TO_MODEL))
		os.mkdir(PATH_TO_MODEL)

	# loading the training data to build the mapper and get the size of the different sets
	print('loading training data ... ')
	mapper, attributes_train, reviews_train_source, reviews_train_target = building_data_train(PATH_TO_DATA, limit= None)
	# create a reverse mapper dictionnary to translate embedding in tokens
	reverse_mapper = {index:token for token, index in mapper.items()}
	# saving the mapper into the model folder 
	with open(os.path.join(PATH_TO_MODEL, 'mapper.json'), 'w') as mapper_file:
		json.dump(mapper, fp = mapper_file)

	# getting useful parameters
	VOCABULARY_SIZE = len(mapper)
	REVIEW_LEN = reviews_train_source.shape[-1]
	ATTRIBUTES_LEN = attributes_train.shape[-1]
	NB_OF_SAMPLES = reviews_train_source.shape[0]

	# loading or creating the model

	# input for attributes
	input_attributes = Input(shape = (ATTRIBUTES_LEN,), name = 'input_attributes')
	# creating an embedding layer for the attributes
	embedding_attributes_layer = Embedding(input_dim = VOCABULARY_SIZE, output_dim = VOCABULARY_SIZE, name = 'embedding_attributes_layer')
	# embedding attributes
	embedded_attributes = embedding_attributes_layer(input_attributes)
	# creating an lstm layer for attributes
	lstm_attributes = LSTM(LSTM_NUMBER_UNITS,return_state = True, name ='lstm_attributes')
	# lstming attributes
	attributes_lstm_outputs, attributes_lstm_h, attributes_lstm_c = lstm_attributes(embedded_attributes)
	# concatenating results to use them in the decoder
	attributes_state = [attributes_lstm_h, attributes_lstm_c]

	# input for reviews
	input_reviews = Input(shape=(REVIEW_LEN,), name = 'input_reviews')
	# creating an embedding layer for the reviews
	embedding_reviews_layer = Embedding(input_dim = VOCABULARY_SIZE, output_dim = VOCABULARY_SIZE, name = 'embedding_reviews_layer')
	# embedding reviews
	embedded_reviews = embedding_reviews_layer(input_reviews)
	# dropping half of the units 
	dropout_reviews = Dropout(.5, name = 'dropout_reviews')(embedded_reviews)
	# creating a lstm layer for reviews
	lstm_reviews = LSTM(LSTM_NUMBER_UNITS, return_sequences=True, return_state = True, name = 'lstm_reviews')
	# lstming the reviews
	reviews_lstm_outputs, _ , _ = lstm_reviews(dropout_reviews, initial_state=attributes_state)
	# creating a dense layer for preparing the outputs
	dense_layer = Dense(VOCABULARY_SIZE,activation='softmax', name = 'output_layer')
	# densing the reviews
	output_reviews = dense_layer(reviews_lstm_outputs)

	# creating the encoder-decoder model
	model = Model(inputs=[input_attributes, input_reviews],outputs=[output_reviews])
	# to carry on learning from the last time 
	if LOAD_MODEL: model.load_weights(os.path.join(PATH_TO_MODEL, 'main_weights.h5'))

	model.summary()

	# defining an optimizer
	opt = RMSprop(lr = .0001)

	# compiling the model
	model.compile(optimizer = opt, loss = ['categorical_crossentropy'], metrics = ['accuracy'])

	# saving the model architecture 
	json_model = model.to_json()
	with open(os.path.join(PATH_TO_MODEL, 'main_architecture.json'), 'w') as model_file:
		model_file.write(json_model)

	# running through the epochs
	for index_epoch in range(1, EPOCHS+1):

		# generating a list of batch indexes randomly
		list_of_batch_index = generate_batches_index(number_of_samples = NB_OF_SAMPLES, batch_size = BATCH_SIZE)

		# running through the batches 
		for id_batch, batch_indexes in enumerate(list_of_batch_index):

			# fitting the model on the batch
			model.fit([attributes_train[batch_indexes], reviews_train_source[batch_indexes]], 
				to_categorical(reviews_train_target[batch_indexes], num_classes = VOCABULARY_SIZE), 
				epochs = 1, verbose = False)

			# printing a prediction to see progress
			if (id_batch % 10 == 0) and not SILENT:
				# making a prediction
				predictions = model.predict([attributes_train[batch_indexes[:1]], reviews_train_source[batch_indexes[:1]]])
				# getting the predicted tokens and turning them into a review
				predictions = np.argmax(predictions, axis = -1)
				review_pred = generate_list_of_tokens(predictions, reverse_mapper)
				review_pred = create_review(review_pred[0])
				# getting the actual review tokens and turning then into a review
				real_review = reviews_train_target[batch_indexes[:1],:]
				review_act = generate_list_of_tokens(real_review, reverse_mapper)
				review_act = create_review(review_act[0])

				# printing the results
				print('epoch {}/{} - batch {}/{}'.format(index_epoch, EPOCHS, id_batch+1, len(list_of_batch_index)))
				print('actual : \n', review_act)
				print('predicted : \n', review_pred)

		# creating the predictive models in order to save them
		# encoder
		predictive_model_enc = Model(inputs = input_attributes, outputs = attributes_state)

		# decoder
		input_hidden = Input(shape=(LSTM_NUMBER_UNITS,))
		input_context = Input(shape=(LSTM_NUMBER_UNITS,))
		# concatenating the two output from the encoder
		predictive_model_dec_input = [input_hidden, input_context]
		# passing through the same lstm layer as during training
		predictive_model_dec_out, predictive_model_dec_hidden, predictive_model_dec_context = lstm_reviews(embedding_reviews_layer(input_reviews), 
		                                                 initial_state=[input_hidden, input_context])
		# concatenating the states of the lstm layer
		# predictive_reviews_states = [predictive_model_dec_hidden, predictive_model_dec_context]
		# applying the dense layer for classification 
		predictive_model_dec_out = dense_layer(predictive_model_dec_out)
		# creating the decoder model
		predictive_model_dec = Model(inputs=[input_reviews, input_hidden, input_context],
		                          outputs=[predictive_model_dec_out, predictive_model_dec_hidden, predictive_model_dec_context])

		
		# saving the weigths from the main model 
		model.save_weights(os.path.join(PATH_TO_MODEL, 'main_weights.h5'))

		# saving the encoder model
		## architecture
		enc_json = predictive_model_enc.to_json()
		with open(os.path.join(PATH_TO_MODEL, 'enc_architecture.json'), 'w') as model_file1:
			json.dump(enc_json, fp = model_file1)
		## weights
		predictive_model_enc.save_weights(os.path.join(PATH_TO_MODEL, 'enc_weights.h5'))

		# saving the decoder model
		## architecture
		dec_json = predictive_model_dec.to_json()
		with open(os.path.join(PATH_TO_MODEL, 'dec_architecture.json'), 'w') as model_file2:
			json.dump(dec_json, fp = model_file2)
		## weights
		predictive_model_dec.save_weights(os.path.join(PATH_TO_MODEL, 'dec_weights.h5'))
		predictive_model_enc.summary()
		predictive_model_dec.summary()


	print('######### Learning done ##########')