# deepConvo: a project for CS 229 at Stanford University during fall 2018.

import keras.backend as K

from keras.layers import Input, Dense, Activation, BatchNormalization, Flatten, Conv3D, MaxPooling3D
from keras.models import Model
from keras.initializers import glorot_uniform

from data_process import get_data

# The number of epochs to train the model.
NUM_EPOCHS = 10

# The number of ordered frames per data point tensor.
NUM_FRAMES_PER_TENSOR = 50

# Dimensions of the input frames.
INPUT_DIM = (NUM_FRAMES_PER_TENSOR, 384, 512, 3)

# Dataset relative path.
DATASET_PATH = '../datasets/VidTIMIT Audio-Video Dataset/data/'

# Training dataset split. Must be in (0, 1).
TRAIN_SPLIT = .95

# Configures Keras.
def setup():
    K.set_image_data_format('channels_last')
    K.set_learning_phase(1)


# Returns a model with the layers described in the function.
def get_model_from_architecture(input_shape, classes):
    bn_name_base = "lipreader"

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

    print 'Gathering data...'
    x_train, y_train, x_test, y_test = get_data(DATASET_PATH, TRAIN_SPLIT, NUM_FRAMES_PER_TENSOR)

    model = get_model_from_architecture(input_shape = INPUT_DIM, classes = 2)
    model.compile(optimizer = 'adam', loss = 'categorical_crossentropy', metrics = ['accuracy'])
    model.fit(x_train, y_train, epochs = 3, batch_size = 32)

    predictions = model.evaluate(x_test, y_test)
    print "Loss = " + str(predictions[0])
    print "Test accuracy = " + str(predictions[1])
