# Video sequence emotion classification with Inception and LSTM:
### Categorizing facial expression into different class of emotion plays an important role in automatic emotions/sentiments
### recognition varying over time. we extracted feature embeddings by using Inceptionv3 architecture and trained it with RNN network topology
### Download Ravdess data: https://zenodo.org/record/1188976#.YPjw45MzbRY

1. Place the videos from your dataset in data/train and data/test folders. Each video type should have its own folder

>	| data/test
> >		| happy
> >		| neutral
> >		...
>	| data/train
> >		| happy
> >		| neutral
> >		...


2. We'll first need to extract images from each of the videos. We'll need to record the following data in the file:
   [train|test], class, filename, nb frames
   
   Goto data folder and run
`
        $ python3 generate_img_files.py mp4`


3. Extract sequential features from images of each class identity

`	$ python3 features_extraction.py`

4. Check the data_file.csv and choose the acceptable sequence length of frames. It should be less or equal to lowest one if you want to process all videos in dataset.
5. Extract sequence for each video with InceptionV3 and train LSTM. Run train.py script with sequence_length, class_limit, image_height, image_width args

`	$ python3 train_split.py 60 8 720 1280`
        Features shape: Number of samples x 60 x 2048
6. Save your best model file. (For example, lstm-features.hdf5)

## Requirements

`pip3 install -r requirements.txt`

