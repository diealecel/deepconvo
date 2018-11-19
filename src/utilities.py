# Encoding: utf-8

# deepConvo: a project for CS 229 at Stanford University during fall 2018.

from sys import stdout
from random import shuffle
from os import listdir
from os.path import join, isdir, isfile

# Prints a progress bar to inform user of work being done.
def print_progress(iteration, total, prefix = '', suffix = '', decimals = 0, bar_length = 50):
    """
    Call in a loop to create terminal progress bar
    @params:
        iteration   - Required  : current iteration (Int)
        total       - Required  : total iterations (Int)
        prefix      - Optional  : prefix string (Str)
        suffix      - Optional  : suffix string (Str)
        decimals    - Optional  : positive number of decimals in percent complete (Int)
        bar_length  - Optional  : character length of bar (Int)
    """
    str_format = "{0:." + str(decimals) + "f}"
    percents = str_format.format(100 * (iteration / float(total)))
    filled_length = int(round(bar_length * iteration / float(total)))
    bar = u'█' * filled_length + '-' * (bar_length - filled_length)

    completed_progress_bar = '\r%s |%s| %s%s %s' % (prefix, bar, percents, '%', suffix)
    stdout.write(completed_progress_bar)

    # Print new line when complete
    if iteration == total: stdout.write('\n')
    stdout.flush()


# Returns the subdirectory names in |path|.
def get_immediate_subdirs(path):
    subdirs = []
    for thing in listdir(path):
        if isdir(join(path, thing)):
            subdirs.append(thing)

    return subdirs


# Returns the file names in |path|.
def get_files(path):
    files = []
    for thing in listdir(path):
        if isfile(join(path, thing)):
            files.append(thing)

    return files


# Returns the subject train and test splits from |subjects| according to |train_split|.
def split_subdirs(subjects, train_split):
    assert train_split > 0 and train_split < 1
    desired_train_num = int(len(subjects) * train_split)

    if desired_train_num == 0 or desired_train_num == len(subjects):
        raise Exception('Cannot allocate enough training and testing examples. Please modify TRAIN_SPLIT.')

    shuffle(subjects)
    train_subjects = subjects[:desired_train_num]
    test_subjects = subjects[desired_train_num:]

    return train_subjects, test_subjects
