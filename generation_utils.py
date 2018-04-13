import numpy as np
import json 
from tqdm import tqdm as prog_bar
import os

def generate_word(reverse_mapper, distribution = None, VOCABULARY_SIZE = 10):
	if distribution is None:
		distribution = np.random.uniform(size = VOCABULARY_SIZE)
	index = np.argmax(distribution)
	token = reverse_mapper[index]
	return token


def create_review(tokens, name = 'Name of the resto', near = None, area = None):
	text = ' '.join(tokens).replace(' .', '.').replace(' ,', ',')
	if name is not None : text = text.replace('<name>', name)
	if near is not None : text = text.replace('<near>', near)
	if area is not None : text = text.replace('<area>', area)
	for null_token in ['<pad>', '<end>', '<start>']:
		text = text.replace(null_token, '')
	return text

if __name__=='__main__':


	mapper = dict(json.load(open(os.path.join('data', 'mapper.json'))))
	reverse_mapper = {int(value)-1: key for key, value in mapper.items()}

	VOCABULARY_SIZE = len(mapper)


	list_of_tokens = []
	for i in range(76):
		list_of_tokens.append(generate_word(reverse_mapper = reverse_mapper, VOCABULARY_SIZE = VOCABULARY_SIZE))

	text = create_review(list_of_tokens)

	print("\n"*3)
	print(text)
	print("\n"*3)
