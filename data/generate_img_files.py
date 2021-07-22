"""
Nitesh Kumar Chaudhary
niteshku001@e.ntu.edu.sg

Once we have videos split between train and test, and
all nested within folders representing their classes. We'll 
first need to extract images from each of the videos. We'll
need to record the following data in the file:

    [train|test], class, filename, nb frames

python3 extract_files.py mp4

"""
import csv
import glob
import os
import os.path
import sys
from subprocess import call

def gen_img_files(extenssion='mp4'):
    data_file = []
    folders = ['train', 'test']

    for folder in folders:
        class_folders = glob.glob(os.path.join("/data/niteshku001/Ravdess/data", folder, '*'))
        print("class_folder: ", class_folders)
        for vid_class in class_folders:
            class_files = glob.glob(os.path.join(vid_class, '*.' + extenssion))

            for video_path in class_files:
                print("video_path: ", video_path)
                # Get the parts of the file.
                video_parts = get_video_parts(video_path)
                print("video_parts: ", video_parts)
                train_or_test, classname, filename_no_ext, filename = video_parts

                # Only extract if we haven't done it yet. Otherwise, just get
                # the info.
                if not check_already_extracted(video_parts):
                    # Now extract it.
                    src = os.path.join("/data/niteshku001/Ravdess/data", train_or_test, classname, filename)
                    dest = os.path.join("/data/niteshku001/Ravdess/data", train_or_test, classname,
                        filename_no_ext + '-%04d.jpg')
                    call(["ffmpeg", "-i", src, dest])

                # Now get how many frames it is.
                nb_frames = get_nb_frames_for_video(video_parts)

                data_file.append([train_or_test, classname, filename_no_ext, nb_frames])

                print("Generated %d frames for %s" % (nb_frames, filename_no_ext))

    with open('data_file.csv', 'w') as fout:
        writer = csv.writer(fout)
        writer.writerows(data_file)

    print("Extracted and wrote %d video files." % (len(data_file)))

def get_nb_frames_for_video(video_parts):
    """Given video parts of an (assumed) already extracted video, return
    the number of frames that were extracted."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    generated_files = glob.glob(os.path.join("/data/niteshku001/Ravdess/data", train_or_test, classname,
                                filename_no_ext + '*.jpg'))
    return len(generated_files)

def get_video_parts(video_path):
    """Given a full path to a video, return its parts."""
    parts = video_path.split(os.path.sep)
    print("parts: ", parts)
    filename = parts[7]
    filename_no_ext = filename.split('.')[0]
    classname = parts[6]
    train_or_test = parts[5]

    return train_or_test, classname, filename_no_ext, filename

def check_already_extracted(video_parts):
    """Check to see if we created the -0001 frame of this file."""
    train_or_test, classname, filename_no_ext, _ = video_parts
    return bool(os.path.exists(os.path.join("/data/niteshku001/Ravdess", train_or_test, classname,
                               filename_no_ext + '-0001.jpg')))

def main():
    
    if (len(sys.argv) == 2):
        gen_img_files(sys.argv[1])
    else:
        print ("Try: python3 generate_img_files.py mp4")

if __name__ == '__main__':
    main()
