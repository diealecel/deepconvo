# deepConvo: a project for CS 229 at Stanford University during fall 2018.

import numpy as np

from os.path import join
from keras.utils.np_utils import to_categorical

from utilities import get_immediate_subdirs, split_subdirs, print_progress
from tensor_generation import generate_rgb_tensor, generate_optical_flow_tensor

# Subdirectory that contains video frames within subject files.
FRAMES_SUBDIR = 'video'

# Used to construct sentence ID dictionary. Ignores classes that contain this word.
REJECT_ID = 'head'

# Returns a sentence map, which takes as input a sentence ID and outputs a
# generated index such that there is a bijection from sentence ID's to indices.
# To generate this sentence map, an |example_subject| in the dataset located at
# |path| is used. This assumes that the subject has a directory named after
# the sentence ID's that are shared among all subjects.
def create_sentence_map(path, example_subject):
    # example_dir = join(path, example_subject, FRAMES_SUBDIR)
    #
    # all_ids = get_immediate_subdirs(example_dir)
    # sentence_ids = [ id for id in all_ids if not id.startswith(REJECT_ID) ]
    # sentence_map = { sentence_ids[i]:i for i in xrange(len(sentence_ids)) }
    sentence_map = {'sa1':0, 'sa2':1}

    return sentence_map


# Returns the tensor made from |data_path| according to |num_frames_per_tensor|
# and |data_type|.
def get_tensor(data_path, num_frames_per_tensor, data_type):
    if data_type == 'rgb':
        return generate_rgb_tensor(data_path, num_frames_per_tensor)
    if data_type == 'optical':
        return generate_optical_flow_tensor(data_path, num_frames_per_tensor)


# Returns data points from |subjects| as numpy tensors prepared according to
# |data_type|. The layers of the tensors are |num_frames_per_tensor| tall, each
# containing one frame of the video. The dataset is found at |path|.
def process_data(subjects, path, num_frames_per_tensor, data_type):
    sentence_map = create_sentence_map(path, subjects[0])

    # For keeping track of progress to be displayed in the progress bar.
    num_tensors = len(subjects) * len(sentence_map)
    tensors_done = 0

    x = []
    y = []
    for subject in subjects:
        for sentence_id in sentence_map:
            data_path = join(path, subject, FRAMES_SUBDIR, sentence_id)
            frame_tensor = get_tensor(data_path, num_frames_per_tensor, data_type)
            x.append(frame_tensor)
            y.append(sentence_map[sentence_id])

            tensors_done += 1
            print_progress(tensors_done, num_tensors, 'Please wait.')

    x = np.stack(x)
    y = to_categorical(np.stack(y))
    return x, y


# Returns numpy tensors as training and testing data according to |train_split|
# and |num_frames_per_tensor| from |path|. These numpy tensors are prepared
# according to |data_type|.
def get_data(path, train_split, num_frames_per_tensor, data_type):
    subjects = get_immediate_subdirs(path)
    train_subjects, test_subjects = split_subdirs(subjects, train_split)

    print 'Processing training data.'
    x_train, y_train = process_data(train_subjects, path, num_frames_per_tensor, data_type)

    print 'Processing testing data.'
    x_test, y_test = process_data(test_subjects, path, num_frames_per_tensor, data_type)

    return x_train, y_train, x_test, y_test
