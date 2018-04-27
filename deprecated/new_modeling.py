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

def parsing_attributes(list_of_attributes):
	tokenized_attributes = []
	print('tokenizing attributes ...')
	for attributes in prog_bar(list_of_attributes):
		# we tokenize the attributes string
		tokenized_attributes.append(['<start>'] + tokenize(attributes))
	pad_len = len(max(tokenized_attributes, key = len))
	padded_attributes = []
	print('padding attributes ...')
	for attributes in prog_bar(tokenized_attributes):
		att_len = len(attributes)
		attributes += ['<pad>']*(pad_len - att_len)
		padded_attributes.append(attributes[:pad_len] + ['<end>'])
	
	return np.array(padded_attributes)


def building_data_train(path, limit = None):
	"""function to build the dictionnary of the data"""
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

	pad_len_rev = len(max(list_of_reviews, key = len))
	pad_len_att = len(max(list_of_attributes, key = len))

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

