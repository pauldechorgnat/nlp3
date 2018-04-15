import numpy as np
import json 
from tqdm import tqdm as prog_bar
import os

def generate_word_stoch(reverse_mapper, distribution = None, VOCABULARY_SIZE = 10):
	if distribution is None:
		distribution = np.random.uniform(size = VOCABULARY_SIZE)
	index = np.argmax(distribution)
	token = reverse_mapper[index]
	return token


def create_review(tokens, name = '<name>', near = None, area = None):
	text = ' '.join(tokens).replace(' .', '.').replace(' ,', ',').replace(' s ', "'s ").replace(' d ', "'d ")
	if name is not None : text = text.replace('<name>', name)
	if near is not None : text = text.replace('<near>', near)
	if area is not None : text = text.replace('<area>', area)
	for null_token in ['<pad>', '<end>', '<start>']:
		text = text.replace(null_token, '')
	return text

def generate_word_det(token, reverse_mapper):
	return reverse_mapper.get(token)

def generate_list_of_tokens(output_predictions, reverse_mapper):
	translated_output = []
	for output_review in output_predictions:
		translated_review = []
		for token in output_review:
			translated_review.append(reverse_mapper.get(token))
		translated_output.append(translated_review)
	return np.array(translated_output)




if __name__=='__main__':


	mapper = dict(json.load(open(os.path.join('data', 'mapper.json'))))
	reverse_mapper = {int(value)-1: key for key, value in mapper.items()}

	VOCABULARY_SIZE = len(mapper)


	list_of_tokens = []
	for i in range(76):
		list_of_tokens.append(generate_word_stoch(reverse_mapper = reverse_mapper, VOCABULARY_SIZE = VOCABULARY_SIZE))

	text = create_review(list_of_tokens)

	print("\n"*3)
	print(text)
	print("\n"*3)
