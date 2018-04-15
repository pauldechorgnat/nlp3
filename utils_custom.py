import tensorflow as tf
import numpy as np

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

	