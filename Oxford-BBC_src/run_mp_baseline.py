# deepConvo: a project for CS 229 at Stanford University during fall 2018.

import keras.backend as K

from math import ceil
from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv3D, MaxPooling3D
from keras.models import Model
from keras.initializers import glorot_uniform

from data_process import OxfordBBCSequence

# The number of workers to spawn for multiprocessing.
NUM_WORKERS = 100

# The number of epochs to train the model.
NUM_EPOCHS = 10

# The percentage of all batches to use for training per epoch.
BATCH_PERCENTAGE = .005

# The batch size to be used for training.
BATCH_SZ = 20

# The number of ordered frames per data point tensor.
NUM_FRAMES_PER_TENSOR = 29

# The number of classes the model will try to discriminate.
NUM_CLASSES = 500

# The number of training examples in the dataset.
NUM_TRAIN_EXAMPLES = 488766

# The number of validation examples in the dataset.
NUM_VAL_EXAMPLES = 25000

# Dimensions of the input frames.
INPUT_DIM = (NUM_FRAMES_PER_TENSOR, 256, 256, 3)

# Dataset relative path.
DATASET_PATH = '/home/diego/Oxford-BBC LRW Dataset'

# The tensor type to use in training and testing.
TENSOR_TYPE = 'rgb'

# The number of words to train and evaluate on
NUM_WORDS = 2

# Configures Keras.
def setup():
    K.set_image_data_format('channels_last')
    K.set_learning_phase(1)


# Returns a model with the layers described in the function.
def get_model_from_architecture(input_shape, classes):
    bn_name_base = 'lipreader'

    # Define the input as a tensor with shape |input_shape|.
    X_input = Input(input_shape)

    X = Conv3D(40, (3, 3, 3), strides = (2, 2, 2), name = 'conv1', kernel_initializer = glorot_uniform(seed = 0))(X_input)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2a')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((2, 2, 2), strides = (2, 2, 2))(X)

    X = Conv3D(2, (1, 1, 1), strides = (2, 2, 2), name = 'conv2', kernel_initializer = glorot_uniform(seed = 0))(X)
    X = BatchNormalization(axis = 3, name = bn_name_base + '2b')(X)
    X = Activation('relu')(X)
    X = MaxPooling3D((1, 2, 2), strides = (2, 2, 2))(X)

    # Output layer.
    X = Flatten()(X)
    X = Dense(classes, activation = 'softmax', name = 'fc' + str(classes), kernel_initializer = glorot_uniform(seed = 0))(X)

    # Create model.
    model = Model(inputs = X_input, outputs = X, name = '3Dlipreader')

    return model


if __name__ == '__main__':
    setup()
    print '\nData mini-batches are iteratively generated to efficiently use RAM.'
    print 'Please note: multiprocessing enabled. Currently running on ' + str(NUM_WORKERS) + ' threads.\n'

    # Set up for training.
    train_batch_generator = OxfordBBCSequence(DATASET_PATH, 'train', BATCH_SZ, NUM_FRAMES_PER_TENSOR, TENSOR_TYPE, NUM_WORDS)
    num_train_batches = int(ceil(1.0 * NUM_TRAIN_EXAMPLES / BATCH_SZ) * BATCH_PERCENTAGE)

    model = get_model_from_architecture(input_shape = INPUT_DIM, classes = NUM_CLASSES)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit_generator(train_batch_generator, epochs = NUM_EPOCHS, steps_per_epoch = num_train_batches, \
                        use_multiprocessing = True, workers = NUM_WORKERS)
    train_batch_generator.log_file(False)
    
    # Set up for testing.
    test_batch_generator = OxfordBBCSequence(DATASET_PATH, 'val', BATCH_SZ, NUM_FRAMES_PER_TENSOR, TENSOR_TYPE, NUM_WORDS)
    num_test_batches = int(ceil(1.0 * NUM_VAL_EXAMPLES / BATCH_SZ) * BATCH_PERCENTAGE)

    predictions = model.evaluate_generator(test_batch_generator, steps = num_test_batches, \
                                           use_multiprocessing = True, workers = NUM_WORKERS)
    print(test_batch_generator.word_trackers)
    test_batch_generator.log_file(False)
    print(train_batch_generator.word_trackers)
    print "Loss = " + str(predictions[0])
    print "Test accuracy = " + str(predictions[1])
