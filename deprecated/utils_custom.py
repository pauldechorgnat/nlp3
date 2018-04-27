import tensorflow as tf
import numpy as np
from collections import Counter

def length(sequence):
  used = tf.sign(tf.reduce_max(tf.abs(sequence), 2))
  length = tf.reduce_sum(used, 1)
  length = tf.cast(length, tf.int32)
  return length


def generate_batches_index(number_of_samples = 42064, batch_size = 64):
	# reordering samples
	indexes = np.random.choice(a = range(number_of_samples), replace = False, size = number_of_samples)
	list_of_batch_index = []
	for i in range(number_of_samples//batch_size):
		list_of_batch_index.append(indexes[i*batch_size:(i+1)*batch_size])
	return list_of_batch_index


def loss_function(predictions, actual):
	# predictions have size (None, PADDING_LENGTH, VOCABULARY_SIZE)
	# actual have size (None, PADDING_LENGTH)
	loss = 0
	for index_sample, sample in enumerate(predictions):
		for index_token, token in enumerate(actual):
			label = actual[index,index_token]
			prediction_probability = predictions[index, index_token, label]
			loss += np.log(prediction_probability)
	return loss

def saving_model_architecture(model, path='model.json'):
	json_model = model.to_json()
	with open(path, 'w') as model_file:
		model_file.write(json_model)
	print('model architecture saved')

def compute_bleu_score(actual_tokens, predicted_tokens):
	actual_counter = Counter(actual_tokens)
	predicted_counter = Counter(predicted_tokens)
	bleu = 0
	for token in predicted_counter.keys():
		bleu += min(predicted_counter.get(token), actual_counter.get(token, 0))
	bleu = bleu / len(actual_tokens)
	return bleu

def compute_rouge_score(actual_tokens, predicted_tokens, n = 1):
	actual_n_grams = [tuple(actual_tokens[i:i+n]) for i in range(len(actual_tokens)-n+1)]
	predicted_n_grams = [tuple(predicted_tokens[i:i+n]) for i in range(len(predicted_tokens)-n+1)]
	rouge = 0
	for n_gram in actual_n_grams:
		if n_gram in predicted_n_grams:
			rouge += 1
	rouge = rouge/len(actual_n_grams) # in our case rouge precision and recall are the same as the length of the predictions are the same
	return rouge


def make_prediction(attributes, encoder, decoder, mapper, review_length, vocabulary_size):
    attributes = np.array(attributes)
    # Initial states value is coming from the encoder 
    print(attributes)
    print(attributes.shape)
    states = encoder.predict(attributes)
    print(states)
    
    target_seq = np.zeros((1,  review_length))
    target_seq[0,  0] = mapper.get('<start>')
    
    predicted_review = []
    stop_condition = False
    
    for i in range(review_length):
        
        decoder_out, decoder_h, decoder_c = decoder.predict(x=[target_seq] + states)
        # getting the index of the token
        index = np.argmax(decoder_out[0,-1,:])
        predicted_review += [index]

        
        target_seq = np.zeros((1,  vocabulary_size))
        target_seq[0,  index] = 1
        
        states_val = [decoder_h, decoder_c]
        
    return predicted_review


if __name__=='__main__':
	actual_tokens = ['the', 'cat', 'is', 'black', 'and', 'grey']
	predicted_tokens = ['the', 'kitten', 'is', 'black', 'and', 'cute']
	print(actual_tokens)
	print(predicted_tokens)
	print('bleu score : ', compute_bleu_score(actual_tokens, predicted_tokens))
	print('rouge score : ', compute_rouge_score(actual_tokens, predicted_tokens, n = 2))

