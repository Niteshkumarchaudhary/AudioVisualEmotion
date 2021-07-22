"""
This script generates extracted features for each video, which other
models make use of.

You can change you sequence length and limit to a set number of classes
below.

class_limit is an integer that denotes the first N classes you want to
extract features from.
Then set the same number when training models.

Nitesh Kumar Chaudhary
niteshku001@e.ntu.edu.sg

"""
import numpy as np
import os.path
from data import DataSet
from extractor import Extractor_inception
from tqdm import tqdm
import pickle

def extract_features(seq_length=60, class_limit=8, image_shape=(720, 1280, 3)):
    # Get the dataset.
    data = DataSet(seq_length=seq_length, class_limit=class_limit, image_shape=image_shape)

    # get the model.
    model = Extractor_inception(image_shape=image_shape)
    video_feat = []
    video_label = []
    # Loop through data.
    pbar = tqdm(total=len(data.data))
    for video in data.data:

        # Get the path to the sequence for this video.
        print("video 2: ", video[2])
        #print("video 2: ", video)
        path = os.path.join('/data/niteshku001/Ravdess/data', 'sequences', video[2] + '-' + str(seq_length) + \
            '-features')  # numpy will auto-append .npy
        label = int(video[2].split("-")[2])-1
        print(label)
        # Check if we already have it.
        
        if os.path.isfile(path + '.npy'):
            pbar.update(1)
            continue
        
        # Get the frames for this video.
        frames = data.get_frames_for_sample(video)
        print("frames: ", np.array(frames).shape)
        # Now downsample to just the ones we need.
        frames = data.rescale_list(frames, seq_length)
        print("frames shape: ", np.array(frames).shape)
        # Now loop through and extract features to build the sequence.
        sequence = []
        for image in frames:
            features = model.extract(image)
            sequence.append(features)

        video_feat.append(sequence)
        video_label.append(label)
        
        print("video features: ", np.array(video_feat).shape)
        print("video label: ", label)
        
        # Save the sequence.
        print("Sequence shape: ", np.array(sequence).shape)
        np.save(path, sequence)

        pbar.update(1)
        
    with open('/data/niteshku001/Ravdess/data/video_features', 'wb') as Features:
        pickle.dump((np.array(video_feat), np.array(video_label)), Features)
    print("Completed the video feature extraction")
    
    pbar.close()

extract_features()

data = DataSet(seq_length=60, class_limit=8, image_shape=(720, 1280, 3))
"""
data_type = 'features'
X, y = data.get_all_sequences_in_memory('train', data_type)
X_test, y_test = data.get_all_sequences_in_memory('test', data_type)
print("Train X: ", X.shape)
print("Train y: ", y.shape)

print("Test X: ", X_test.shape)
print("Test y: ", y_test.shape)
"""


