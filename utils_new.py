# importing libraries 
import os
import re
import numpy as np
from collections import Counter
from tqdm import tqdm as prog_bar

def tokenize(text):
	"""own personnal tokenizer function"""
	pattern = '(\w+)'
	splits = re.split(string = text, pattern = pattern)
	flags = {'startofreview': '<start>', 'endofreview':'<end>', 'nameplaceholder':'<name>', 'nearplaceholder':'<near>', 'areaplaceholder':'<area>'}		
	clean_splits = [val.replace(" ", "") if val not in flags.keys() else flags.get(val) for val in splits ]
	return [val for val in clean_splits if len(val)>0]


def building_data_train(path, padding_length_att = 100, padding_length_rev = 100, limit = None):
	"""function to build the format the training data"""
	list_of_reviews = []
	list_of_attributes = []

	# opening the file
	with open(path, 'r', encoding = 'utf-8') as file:
	    for line_number, line in prog_bar(enumerate(file)):
	        if line_number==0: continue # skipping the header line
	        if '[' not in str(line): continue # some lines are fuzzy 
	        if (limit is not None) and line_number > limit : break
	        attributes = str(line).split('",')[0].replace('"', '') # cosmetic
	        review = str(line).split('",')[1].replace('"', '').replace('\n', '') # cosmetic
	        # tokenizing the data
	        review = tokenize(review)
	        attributes = tokenize(attributes)

	        # appending the tokenized reviews and attributes to the output lists
	        list_of_reviews.append(review)
	        list_of_attributes.append(attributes)

	pad_len_rev = padding_length_rev # len(max(list_of_reviews, key = len))
	pad_len_att = padding_length_att # len(max(list_of_attributes, key = len))

	# padding attributes
	padded_attributes = []
	print('padding attributes ...')
	for attributes in prog_bar(list_of_attributes):
		att_len = len(attributes)
		attributes += ['<pad>']*(pad_len_att - att_len)
		padded_attributes.append(['<start>'] +  attributes[:pad_len_att] + ['<end>'])
	
	# padding the reviews 
	padded_reviews = []
	print('padding reviews ...')
	for review in prog_bar(list_of_reviews):
		rev_len = len(review)
		review += ['<pad>'] * (pad_len_rev-rev_len) 
		padded_reviews.append(['<start>'] +  review[:pad_len_rev]+['<end>'])
	# building vocabularies
	vocabulary_attributes = set([token for att in padded_attributes for token in att])
	vocabulary_reviews = set([token for review in padded_reviews for token in review])
	mapper = vocabulary_reviews.union(vocabulary_attributes)
	mapper = {token:index for index, token in enumerate(mapper)}
	mapper['<unk>'] = len(mapper)
	# translating attributes
	print('translating attributes ...')
	translated_attributes = []
	for att in prog_bar(padded_attributes):
		translated_att = []
		for token in att:
			translated_att.append(mapper.get(token))
		translated_attributes.append(translated_att)
	# translating reviews
	print('translating reviews ...')
	translated_reviews = []
	for review in prog_bar(padded_reviews):
		translated_rev = []
		for token in review :
			translated_rev += [mapper[token]]
		translated_reviews.append(translated_rev)
	target_reviews = [[mapper.get('<pad>')] + translated_rev for translated_rev in translated_reviews]
	source_reviews = [translated_rev + [mapper.get('<pad>')] for translated_rev in translated_reviews]


	return mapper, np.array(translated_attributes), np.array(source_reviews), np.array(target_reviews)


def building_data_test(path, mapper, padding_length_att = 100,  limit = None):
	"""function to build the format the training data"""
	list_of_attributes = []

	# opening the file
	with open(path, 'r', encoding = 'utf-8') as file:
	    for line_number, line in prog_bar(enumerate(file)):
	        if line_number==0: continue # skipping the header line
	        if '[' not in str(line): continue # some lines are fuzzy 
	        if (limit is not None) and line_number > limit : break
	        attributes = str(line).split('",')[0].replace('"', '') # cosmetic
	        # tokenizing the data
	        attributes = tokenize(attributes)

	        # appending the tokenized reviews and attributes to the output lists

	        list_of_attributes.append(attributes)

	pad_len_att = padding_length_att # len(max(list_of_attributes, key = len))

	# padding attributes
	padded_attributes = []
	print('padding attributes ...')
	for attributes in prog_bar(list_of_attributes):
		att_len = len(attributes)
		attributes += ['<pad>']*(pad_len_att - att_len)
		padded_attributes.append(['<start>'] +  attributes[:pad_len_att] + ['<end>'])
	
	
	# translating attributes
	print('translating attributes ...')
	translated_attributes = []
	for att in prog_bar(padded_attributes):
		translated_att = []
		for token in att:
			translated_att.append(mapper.get(token, mapper.get('<unk>')))
		translated_attributes.append(translated_att)
	


	return np.array(translated_attributes)

def generate_list_of_tokens(output_predictions, reverse_mapper):
	"""function to generate tokens out of the reverse_mapper"""
	translated_output = []
	for output_review in output_predictions:
		translated_review = []
		for token in output_review:
			translated_review.append(reverse_mapper.get(token))
		translated_output.append(translated_review)
	return np.array(translated_output)

def create_review(tokens, name = '<name>', near = None, area = None):
	"""function to create a string review out of a list of tokens"""
	text = ' '.join(tokens).replace(' .', '.').replace(' ,', ',').replace(' s ', "'s ").replace(' d ', "'d ")
	if name is not None : text = text.replace('<name>', name)
	if near is not None : text = text.replace('<near>', near)
	if area is not None : text = text.replace('<area>', area)
	for null_token in ['<pad>', '<end>', '<start>']:
		text = text.replace(null_token, '')
	return text


def make_prediction(attributes, encoder, decoder, mapper, review_length, vocabulary_size):
	"""function to make predictions"""

	attributes = np.array(attributes)
	attributes = attributes.reshape(1, -1)
	# Initial states value is coming from the encoder 

	encoder_states = encoder.predict(attributes)

	target_seq = np.zeros((1, review_length+3))
	target_seq[0,0] = mapper.get('<start>')

	review = []
	for i in range(review_length):
		output_decoder, hidden, context = decoder.predict(x= [target_seq, encoder_states[0], encoder_states[1]])
		# getting the index of the token
		token = np.argmax(output_decoder[0,i,:], axis = -1)
		review.append(token)


		target_seq = np.zeros((1, review_length+3))
		target_seq[0,  i] = token

		encoder_states = [hidden, context]

	return review


def generate_batches_index(number_of_samples = 42064, batch_size = 64):
	# reordering samples
	indexes = np.random.choice(a = range(number_of_samples), replace = False, size = number_of_samples)
	list_of_batch_index = []
	for i in range(number_of_samples//batch_size):
		list_of_batch_index.append(indexes[i*batch_size:(i+1)*batch_size])
	return list_of_batch_index

