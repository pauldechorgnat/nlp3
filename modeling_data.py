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

def building_data(path):
	"""function to build the dictionnary of the data"""
	list_of_reviews = []
	list_of_attributes = []
	# opening the file
	with open(path, 'r', encoding = 'utf-8') as file:

	    for line_number, line in prog_bar(enumerate(file)):
	        if line_number==0: continue # skipping the header line
	        if '[' not in str(line): continue # some lines are fuzzy 
	        attributes = str(line).split('",')[0].replace('"', '') # cosmetic
	        review = str(line).split('",')[1].replace('"', '').replace('\n', '') # cosmetic
	        
	        # replace name, near and area by placeholders
	        if 'name' in attributes : 
	            name = attributes.split('name[')[1].split(']')[0]
	            review = review.replace(name, 'nameplaceholder')
	        if 'area' in attributes : 
	            area = attributes.split('area[')[1].split(']')[0]
	            review = review.replace(area, 'areaplaceholder')
	        if 'near' in attributes : 
	            near = attributes.split('near[')[1].split(']')[0]
	            review = review.replace(near, 'nearplaceholder')
	        review = 'startofreview ' + review + ' endofreview'

	        list_of_reviews.append(review.lower())
	        list_of_attributes.append(attributes)

	vocabulary_full = []
	reviews_tokens = []
	print('tokenizing data ...')

	for sentence in prog_bar(list_of_reviews):
	    tokens = tokenize(sentence)
	    vocabulary_full += tokens
	    reviews_tokens.append(tokens)
	
	vocabulary = Counter(vocabulary_full)
	vocabulary = sorted(vocabulary, key = vocabulary.get, reverse = True)
	mapper = {token:index+1 for index, token in prog_bar(enumerate(vocabulary))}
	mapper['<unk>'] = len(mapper)+1
	mapper['<pad>'] = len(mapper)+1

	return mapper, list_of_attributes, reviews_tokens

def building_data_test(path):
    """function to build the dictionnary of the data"""
    print('loading text ...')
    list_of_attributes = []
    # opening the file
    with open(path, 'r', encoding = 'utf-8') as file:

        for line_number, line in prog_bar(enumerate(file)):
            if line_number==0: continue # skipping the header line
            if '[' not in str(line): continue # some lines are fuzzy 
            attributes = str(line).split('",')[0].replace('"', '').replace('\n', '') # cosmetic

            # replace name, near and area by placeholders
            if 'name' in attributes : 
                name = attributes.split('name[')[1].split(']')[0]
                
            if 'area' in attributes : 
                area = attributes.split('area[')[1].split(']')[0]
                
            if 'near' in attributes : 
                near = attributes.split('near[')[1].split(']')[0]
                

            list_of_attributes.append(attributes)

        return list_of_attributes 





def parsing_attributes(attribute):
    # defining dictionnaries to change data into integers and then arrays
    food_dict = {'Chinese': 4, 'English': 1, 'Fast food': 5, 'French': 2, 'Indian': 3, 'Italian': 6, 'Japanese': 0}
    eatType_dict = {'coffee shop': 0, 'pub': 1, 'restaurant': 2}
    ratings_dict = {'1 out of 5': 0, '3 out of 5': 1, '5 out of 5': 2, 'average': 1, 'high': 2, 'low': 0}
    family_dict = {'yes':1, 'no':0}
    price_dict = {'less than £20':0, 'high':2, '£20-25':1, 'moderate':1, 'more than £30':2, 'cheap':0}
    # defining the default arrays 
    food_array = np.zeros((1, 7))
    eatType_array = np.zeros((1, 3))
    ratings_array = np.zeros((1, 3))
    family_array = np.zeros((1, 2))
    price_array = np.zeros((1, 3))
    near_array = np.zeros((1, 1))
    area_array = np.zeros((1,1))
    name_array = np.ones((1,1))
    # splitting the list of attributes according to commas to get the different attributes and their respective values
    splits = attribute.split(',')
    # getting attributes names
    attribute_keys = [att.split('[')[0].replace(' ', '') for att in splits]
    # getting attributes values
    attribute_values = [att.split('[')[1].replace(']', '') for att in splits]
    # creating a dictionnary
    dictionnary = dict(zip(attribute_keys, attribute_values))
    # filling missing attributes with None values
    for att in ['name', 'eatType', 'food', 'priceRange', 'customerrating', 'area', 'familyFriendly', 'near']:
        dictionnary[att] = dictionnary.get(att, None)
    
    # translating data
    if dictionnary['food'] is not None : food_array[0,food_dict[dictionnary['food']]] = 1
    if dictionnary['eatType'] is not None : eatType_array[0,eatType_dict[dictionnary['eatType']]] = 1
    if dictionnary['customerrating'] is not None : ratings_array[0,ratings_dict[dictionnary['customerrating']]] = 1
    if dictionnary['familyFriendly'] is not None : family_array[0,family_dict[dictionnary['familyFriendly']]] = 1
    if dictionnary['priceRange'] is not None : price_array[0,price_dict[dictionnary['priceRange']]] = 1
    if dictionnary['near'] is not None:
        near_array[0,0] = 1
    if dictionnary['area'] is not None:
        area_array[0,0] = 1
    
    # stacking data
    dummy_array = np.hstack([name_array, near_array, area_array, food_array, eatType_array, ratings_array, family_array, price_array])
    
    return dummy_array

def parsing_list_of_attributes(list_of_attributes):
	return_array = [parsing_attributes(att) for att in prog_bar(list_of_attributes)]
	return_array = np.array(return_array)#.reshape(len(list_of_attributes), -1)
	return return_array

def repeating_data(list_of_attributes, sequence_len = 76):
	array = np.repeat(list_of_attributes, sequence_len, axis = 1)
	return array

def padding_review(list_of_reviews, padding_length):
	"""review is a list of tokens"""
	list_of_padded_reviews = []
	for review in prog_bar(list_of_reviews):
		review_len = len(review)
		if review_len < padding_length:
			to_pad = ['<pad>' for _ in range(padding_length - len(review))]
		else : 
			to_pad = []
		padded_review = review + to_pad
		list_of_padded_reviews.append(padded_review[:padding_length])
	return np.array(list_of_padded_reviews)


def mapping_reviews(list_of_reviews, mapper, review_len = 76):
	nb_of_samples = len(list_of_reviews)
	mapped_tokens = []

	for review in prog_bar(list_of_reviews):
		mapped_review = []
		for token in review:
			mapped_review.append(mapper.get(token, mapper.get('<unk>'))-1)
		mapped_tokens.append(mapped_review)
	mapped_tokens = np.array(mapped_tokens, ndmin = 2).reshape(nb_of_samples, review_len)
	return mapped_tokens	