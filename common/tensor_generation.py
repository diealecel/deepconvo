# deepConvo: a project for CS 229 at Stanford University during fall 2018.

import numpy as np

from cv2 import imread, cvtColor, calcOpticalFlowFarneback, cartToPolar, normalize
from cv2 import COLOR_RGB2GRAY, COLOR_RGB2HSV, NORM_MINMAX, COLOR_HSV2RGB
from copy import deepcopy
from deprecated import deprecated

# Used to normalize pixel values.
MAX_PIX_VAL = 255

@deprecated
# Returns the tensor that corresponds to the video frames |frames|. Uses
# |num_frames_per_tensor| to configure the number of frames the tensor uses.
# Note that these frames are stacked chronologically.
def generate_tensor_with_first_frames(frames, num_frames_per_tensor):
    tensor_frames = []
    for i in xrange(num_frames_per_tensor):
        tensor_frames.append(frames[i])

    tensor_frames = np.stack(tensor_frames)
    return tensor_frames


# Returns the tensor that corresponds to the video frames |frames|. Uses
# |num_frames_per_tensor| to configure the number of frames the tensor uses.
# Note that these frames are stacked chronologically and are dispersed as
# uniformly as possible so as to cover the full duration of the video. Also
# note that each frame is normalized.
def generate_rgb_tensor(frames, num_frames_per_tensor):
    if len(frames) < num_frames_per_tensor:
        raise Exception('Not enough frames to generate tensors. Please decrease |NUM_FRAMES_PER_TENSOR|.')

    tensor_frames = []
    uniform_dispersion = np.linspace(0, len(frames) - 1, num = num_frames_per_tensor)
    for i in uniform_dispersion:
        frame = frames[int(i)] / MAX_PIX_VAL
        tensor_frames.append(frame)

    tensor_frames = np.stack(tensor_frames)
    return tensor_frames


# Returns the tensor that corresponds to the video frames |frames| after going
# through RGB optical flow processing. Uses |num_frames_per_tensor| to configure
# the number of frames the tensor uses. Note that these frames are stacked
# chronologically and are dispersed as uniformly as possible so as to cover the
# full duration of the video.
def generate_rgb_optical_flow_tensor(frames, num_frames_per_tensor):
    # Plus one since we need an additional frame to perform optical flow.
    if len(frames) < num_frames_per_tensor + 1:
        raise Exception('Not enough frames to generate tensors. Please decrease |NUM_FRAMES_PER_TENSOR|.')

    tensor_frames = []
    prev_frame = None
    uniform_dispersion = np.linspace(0, len(frames) - 1, num = num_frames_per_tensor + 1)
    for i in uniform_dispersion:
        frame = frames[int(i)]

        # Create initial frame.
        if int(i) == 0:
            prev_frame = cvtColor(frame, COLOR_RGB2GRAY)
            continue

        # Calculate optical flow.
        curr_frame = cvtColor(frame, COLOR_RGB2GRAY)
        flow = calcOpticalFlowFarneback(prev_frame, curr_frame, None, 0.5, 3, 15, 3, 5, 1.2, 0)
        magnitude, angle = cartToPolar(flow[..., 0], flow[..., 1])

        # Construct optical flow representation.
        hsv = np.zeros_like(frame)
        # Set saturation.
        hsv[..., 1] = cvtColor(frame, COLOR_RGB2HSV)[..., 1]
        # Set hue, which corresponds to direction.
        hsv[..., 0] = angle * (180. / np.pi / 2)
        # Set value, which corresponds to magnitude.
        hsv[..., 2] = normalize(magnitude, None, 0, MAX_PIX_VAL, NORM_MINMAX)
        # Convert HSV to RGB.
        rgb_flow = cvtColor(hsv, COLOR_HSV2RGB)

        tensor_frames.append(deepcopy(rgb_flow))
        prev_frame = curr_frame

    tensor_frames = np.stack(tensor_frames)
    return tensor_frames
