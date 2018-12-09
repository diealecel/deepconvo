# deepConvo: a project for CS 229 at Stanford University during fall 2018.

import numpy as np

from os.path import join
from random import randrange, seed, sample
from cv2 import VideoCapture
from keras.utils.np_utils import to_categorical
from keras.utils import Sequence

from common.utilities import get_immediate_subdirs, get_files, print_progress
from common.tensor_generation import generate_rgb_optical_flow_tensor, generate_rgb_tensor

DATA_EXT = '.mp4'

# Returns word trackers as 0 per word in |words|. Trackers are used to count how
# many times a particular word was chosen as part of minibatches.
def init_word_trackers(words):
    return { word:0 for word in words }


# Returns word limits per word in |words| from |dataset_path| according to
# |dataset_path|. A word limit corresponds to how many examples exist for a
# given word, and it is used to determine whether there are any more examples
# under a particular word to use via the word trackers.
def get_word_limits(words, batch_type, dataset_path):
    word_limits = {}

    words_indexed = 0
    for word in words:
        data_path = join(dataset_path, word, batch_type)
        files = get_files(data_path, DATA_EXT)
        word_limits[word] = len(files)

        words_indexed += 1
        print_progress(words_indexed, len(words), 'Indexing dataset.')

    return word_limits

# Initializes and returns the word trackers and word limits that correspond to
# the data at |dataset_path| and |batch_type|.
def init_word_tools(batch_type, dataset_path, limited_words=0):
    words = get_immediate_subdirs(dataset_path)
    if limited_words:
        seed(10)
        words = sample(words, limited_words)

    word_trackers = init_word_trackers(words)
    word_limits = get_word_limits(words, batch_type, dataset_path)

    return word_trackers, word_limits


# Returns a word map, which takes as input a word and outputs a generated index
# such that there is a bijection from words to indices. This assumes that
# |dataset_path| has a folder for each word, which is labeled after the word.
def create_word_map(dataset_path):
    words = get_immediate_subdirs(dataset_path)
    word_map = { words[i]:i for i in xrange(len(words)) }

    return word_map


# Returns a word whose samples have not all yet been explored according to
# |word_trackers| and |word_limits|. Returns None if there are no more words.
def get_valid_word(word_trackers, word_limits):
    words = word_trackers.keys()

    while len(words) > 0:
        rand_i = randrange(len(words))
        rand_word = words[rand_i]
        if word_trackers[rand_word] < word_limits[rand_word]:
            word_trackers[rand_word] += 1
            return rand_word
        else:
            del words[rand_i]

    return None


# Returns the ordered frames of the video at |datum_path|.
def get_frames(datum_path):
    video_capture = VideoCapture(datum_path)

    frames = []
    success, frame = video_capture.read()
    while success:
        frames.append(frame)
        success, frame = video_capture.read()

    return frames


# Returns the tensor made from |datum_path| according to |num_frames_per_tensor|
# and |tensor_type|.
def get_tensor(datum_path, num_frames_per_tensor, tensor_type):
    frames = get_frames(datum_path)

    if tensor_type == 'rgb':
        return generate_rgb_tensor(frames, num_frames_per_tensor)
    if tensor_type == 'rgb_optical_flow':
        return generate_rgb_optical_flow_tensor(frames, num_frames_per_tensor)


# A generator object that generates tuples of examples and labels of size
# |batch_size| from |dataset_path| according to |num_frames_per_tensor| and
# |batch_type|. Tensors are prepared according to |tensor_type|.
def generate_batch(dataset_path, batch_type, batch_size, num_frames_per_tensor, tensor_type):
    word_map = create_word_map(dataset_path)
    word_trackers, word_limits = init_word_tools(batch_type, dataset_path)

    num_examples = sum( word_limits[word] for word in word_limits )
    processed_examples = 0
    while processed_examples < num_examples:
        x = []
        y = []
        for i in range(batch_size):
            word = get_valid_word(word_trackers, word_limits)
            if word is None: break

            datum_filename = word + ('_%05d' % word_trackers[word]) + DATA_EXT
            datum_path = join(dataset_path, word, batch_type, datum_filename)
            frame_tensor = get_tensor(datum_path, num_frames_per_tensor, tensor_type)

            x.append(frame_tensor)
            y.append(word_map[word])

        processed_examples += len(x)
        x = np.stack(x)
        y = to_categorical(np.stack(y), num_classes = len(word_map))

        yield x, y


# A class that emulates generate_batch() translated into a Keras Sequence
# object, made to take advantage of multiprocessing capabilities.
# NOTE: although there is little literature on the inner workings of
#       Keras multiprocessing, this class is believed to be thread-safe. That
#       said, there is a chance that some examples may be used more than once
#       to train in the same epoch. Thankfully, the dataset is so large that
#       the random sampling aspect of the mini-batch generation makes it
#       extremely unlikely for this event to occur. And, if it does, then
#       there are many more data points to offset the bias caused by training on
#       the same data point more than once.
class OxfordBBCSequence(Sequence):

    def __init__(self, dataset_path, batch_type, batch_size, num_frames_per_tensor, tensor_type, limited_words):
        self.dataset_path, self.batch_type, self.tensor_type = dataset_path, batch_type, tensor_type
        self.num_frames_per_tensor, self.batch_size = num_frames_per_tensor, batch_size
        self.word_trackers, self.word_limits = init_word_tools(batch_type, dataset_path, limited_words)
        self.word_map = create_word_map(dataset_path)

        file_name = self.batch_type + "_" + self.tensor_type +  "_logfile.txt"
        self.logfile_path = join(self.dataset_path, '..', file_name)

        self.log_file(True)

    def log_file(self, init):
        with open(self.logfile_path, 'a') as f:
            if init:
                f.write("Selected Dictionary\n\n")
            else:
                f.write("\nExample Counts\n")

            f.write('\n'.join(["%d: %s" % (self.word_map[word], word) for word in self.word_trackers.keys()]))
            f.close()

    def __len__(self):
        num_examples = sum( self.word_limits[word] for word in self.word_limits )
        return num_examples

    def __getitem__(self, idx):
        x = []
        y = []
        for i in range(self.batch_size):
            word = get_valid_word(self.word_trackers, self.word_limits)
            if word is None: break

            datum_filename = word + ('_%05d' % self.word_trackers[word]) + DATA_EXT
            datum_path = join(self.dataset_path, word, self.batch_type, datum_filename)
            frame_tensor = get_tensor(datum_path, self.num_frames_per_tensor, self.tensor_type)

            x.append(frame_tensor)
            y.append(self.word_map[word])

        x = np.stack(x)
        y = to_categorical(np.stack(y), num_classes = len(self.word_map))

        return x, y
