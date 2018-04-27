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

if __name__ == '__main__':
	parser = argparse.ArgumentParser(description='Language generation')

	parser.add_argument('--test_dataset', help= 'Path to the test set', default ='data/testset.csv', type = str )
	parser.add_argument('--path_to_mapper', help= 'Path to mapper', default ='mapper.json', type = str )
	parser.add_argument('--path_to_model', help= 'Path to the model folder', default ='model', type = str )
	parser.add_argument('--path_to_output', help= 'Path to the output of the model', default ='data/output.csv', type = str )

	args = parser.parse_args()

	PATH_TO_DATA = args.test_dataset
	PATH_TO_MODEL = args.path_to_model
	PATH_TO_OUTPUT = args.path_to_output
	PATH_TO_MAPPER = args.path_to_mapper
	# loading mapper
	print('loading mapper ... ')
	with open(os.path.join(PATH_TO_MODEL, PATH_TO_MAPPER), 'r') as mapper_file:
		mapper = dict(json.load(mapper_file))
	# create a reverse mapper dictionnary to translate embedding in tokens
	reverse_mapper = {index:token for token, index in mapper.items()}
	# loading and formating test data
	print('loading test data ...')
	attributes_test = building_data_test(PATH_TO_DATA, mapper = mapper, limit = None)

	print('loading models')
	with open(os.path.join(PATH_TO_MODEL, 'enc_architecture.json'), 'r') as file1 : 
		enc = model_from_json(str(json.load(file1)))
	with open(os.path.join(PATH_TO_MODEL, 'dec_architecture.json'), 'r') as file2 :
		dec = model_from_json(str(json.load(file2)))
	
	enc.load_weights(os.path.join(PATH_TO_MODEL, 'enc_weights.h5'))
	dec.load_weights(os.path.join(PATH_TO_MODEL, 'dec_weights.h5'))
	print("####### ENCODER #######")
	enc.summary()
	print("####### DECODER #######")
	dec.summary()


	print('predicting the reviews ...')
	predictions = [make_prediction(att, encoder = enc, decoder = dec, mapper = mapper, review_length= 100, vocabulary_size= len(mapper)) for att in prog_bar(attributes_test)]
	print(predictions)
	print('cleaning the reviews ...')
	reviews = generate_list_of_tokens(predictions, reverse_mapper)
	reviews = [create_review(rev) for rev in reviews]

	print(reviews)

	with open(PATH_TO_OUTPUT, 'w') as output_file:
		for rev in reviews :
			output_file.write(rev + '\n')


