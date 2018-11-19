# deepConvo: a project for CS 229 at Stanford University during fall 2018.

import numpy as np

from cv2 import imread
from deprecated import deprecated
from os.path import join
from keras.utils.np_utils import to_categorical

from utilities import *

# Subdirectory that contains video frames within subject files.
FRAMES_SUBDIR = 'video'

# Used to construct sentence ID dictionary. Ignores classes that contain this word.
REJECT_ID = 'head'

# Used to normalize input images.
MAX_PIX_VAL = 255

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


@deprecated
# Returns the tensor that corresponds to the video frames at |path|. Uses
# |num_frames_per_tensor| to configure the number of frames the tensor
# uses. Note that these frames are stacked chronologically.
def generate_tensor_with_first_frames(path, num_frames_per_tensor):
    frames = get_files(path)
    frames.sort()

    tensor_frames = []
    for i in xrange(num_frames_per_tensor):
        frame_path = join(path, frames[i])
        frame = imread(frame_path)
        tensor_frames.append(frame)

    tensor_frames = np.stack(tensor_frames)
    return tensor_frames


# Returns the tensor that corresponds to the video frames at |path|. Uses
# |num_frames_per_tensor| to configure the number of frames the tensor uses.
# Note that these frames are stacked chronologically and are dispersed as
# uniformly as possible so as to cover the full duration of the video frames.
# Also note that each frame is normalized.
def generate_tensor(path, num_frames_per_tensor):
    frames = get_files(path)
    frames.sort()

    if len(frames) < num_frames_per_tensor:
        raise Exception('Not enough frames to generate tensors. Please decrease |NUM_FRAMES_PER_TENSOR|.')

    tensor_frames = []
    uniform_dispersion = np.linspace(0, len(frames) - 1, num = num_frames_per_tensor)
    for i in uniform_dispersion:
        frame_path = join(path, frames[int(i)])
        frame = imread(frame_path) / MAX_PIX_VAL
        tensor_frames.append(frame)

    tensor_frames = np.stack(tensor_frames)
    return tensor_frames


# Returns data points from |subjects| as numpy tensors. The layers of the
# tensors are |num_frames_per_tensor| tall, each containing one frame of the
# video. The dataset is found at |path|.
def process_data(subjects, path, num_frames_per_tensor):
    sentence_map = create_sentence_map(path, subjects[0])

    # For keeping track of progress to be displayed in the progress bar.
    num_tensors = len(subjects) * len(sentence_map)
    tensors_done = 0

    x = []
    y = []
    for subject in subjects:
        for sentence_id in sentence_map:
            data_path = join(path, subject, FRAMES_SUBDIR, sentence_id)
            frame_tensor = generate_tensor(data_path, num_frames_per_tensor)
            x.append(frame_tensor)
            y.append(sentence_map[sentence_id])

            tensors_done += 1
            print_progress(tensors_done, num_tensors, 'Please wait.')

    x = np.stack(x)
    y = to_categorical(np.stack(y))
    return x, y


# Returns numpy tensors as training and testing data according to |train_split|
# and |num_frames_per_tensor| from |path|.
def get_data(path, train_split, num_frames_per_tensor):
    subjects = get_immediate_subdirs(path)
    train_subjects, test_subjects = split_subdirs(subjects, train_split)

    print 'Processing training data.'
    x_train, y_train = process_data(train_subjects, path, num_frames_per_tensor)

    print 'Processing testing data.'
    x_test, y_test = process_data(test_subjects, path, num_frames_per_tensor)

    return x_train, y_train, x_test, y_test
