# Natural Language Generation

*** 
### Presentation of our work

For this project, we have tried different things. 

First we wanted to implement an attention based encoder-decoder using tensorflow as it seemed the state-of-the-art technology regarding this competition. Unfortunately, our level in tensorflow did not permit this. We have spent some time on this part but when we figured out that we did not have the time to learn tensorflow and to understand the complex architecture of this model, we decided to switch to keras. 

We then have tried to use the attributes dictionnary of the training set to change them into an array of state and to use an encoder-decoder. Seeing our lack of success, we limited ourself to two bidirectionnal lstm layers stacked with a dense layer. We could see that is was able to learn some things but we were satisfied with the results as it was more likely overfitting the data. (this project is provided as secondary project)

Lately, by speaking with other students, we realized that encoding the attributes was not necessary. We could use it as an input sentence in a Machine Translation algorithm. This is our main output for this project and we will describe it more in details.

### Our project

We have coded with keras an encoder-decoder model that will take as input the attributes and the reviews tokenized and outputs a delayed review. We did not use bidirectionnal lstm for this project although it is mainly used today. 

##### Data preprocessing 

The attributes are tokenized, padded to a given length and mapped to integers using a dictionnary containing every tokens from attributes and reviews. The reviews are processed in the same way.

##### Architecture 

The size of the lstm layer is a parameter of our algorithm but we have chosen it to be 100 by default (for computationnal reasons). The optimizer is a RMSprop with a low learning rate. We are doing some mini-batch learning on a default number of epochs of 100. This allows us to print some early predictions. Weights and architecture are saved in a different folder

##### Command line options

__learn_model.py__
* --path_to_mapper : path to the saving file for the mapper (ie the dictionnary mapping tokens to integers) - default is 'mapper.json'
* --paht_to_model : path to a folder that will contain the model data (if not existing it will be created) - default is 'model'
* --nepochs : number of epochs to perform during training - default is 100
* --lstm_units : number of lstm_units in the model - default is 100
* --silent : boolean that permits to print early predictions - default is True
* --batch_size : size of the minibatch for training - default is 10
* --load_model : boolean value that indicates if we use pretrained weights to initialize the model - default is False (one has to be carful with this one)
__test_model.py__

* --test_dataset : path to the test set - default is 'data/testset.csv'
* --path_to_model : path to the mapper - default is 'mapper.json'
* --path_to_model : path to a folder containing the model data - default is 'model'
* --path_to_output : path to the output file where the reviews are stored - defautl is 'data/output.csv'

_if the code is stored in a folder and the data is in a subfolder named data, the code can run prefectly fine without any command line options_

##### Remarks

* we have used a package called tqdm for visualization of the progress of the algorithm.
* our project is composed of three files : 'learn_model.py', 'test_model.py' and 'utils_new.py'. the last one contains a bunch of custom function that we use, in particular to preprocess the data and postprocess the reviews.

***

### secondary project

#### Command line options

python main_definitif.py
several parameters can be specified, they can be found in the first lines of main_definitif.py: 
* --mode : mode of the algorithm - can be 'train', 'dev' or 'test' - default is 'train'
* --path_to_train : path to the training set - default is 'data/trainset.csv'
* --path_to_dev : path to the dev set - defautl is 'data/devset.csv'
* --path_to_test : path to the test set - default is 'data/testset.csv'
* --pretrained : using a pretrained model (y/...) - default is 'y'
* --path_to_weights  : path to the file containing the weights of the model - default is 'model_weights.h5'
* --path_to_arch : path to the file containing the architecture of the model - default is 'model_architecture.json'
* --path_to_output : path to the file where the output are written in test mode - default is 'data/output_test.csv'
* --silent : printing early predictions (y/...) - default is 'y'

#### Remarks
* this project is using several files : 'utils_custom.py', 'generation_utils.py', 'new_modelling.py' and 'main.py' 
