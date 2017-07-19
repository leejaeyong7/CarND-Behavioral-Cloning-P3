# import definitions
import csv
import cv2

from os import path
import numpy as np

__dirname = path.abspath('.')
_LOCAL_LOG_FILE_PATH= './metadata/driving_log.csv'
_LOCAL_IMAGE_PATH= './metadata/IMG'

# define directory variables
log_file_path = path.normpath(path.join(__dirname,_LOCAL_LOG_FILE_PATH))
image_directory = path.normpath(path.join(__dirname,_LOCAL_IMAGE_PATH))

def setAbsoluteImageDirectory(target_prefix, original_path):
    """
    returns absolute directory given directory, and file path
    """
    # extract filename
    file_name = path.basename(original_path)
    return path.join(image_directory,file_name)

def parseLogLine(line_string):
    """
    parses each line of csv file and returns tuple with size of 7
    """
    line_contents = line_string.split(',')
    center_image_path = setAbsoluteImageDirectory(image_directory,
                                                  line_contents[0])
    left_image_path = setAbsoluteImageDirectory(image_directory,
                                                line_contents[1])
    right_image_path = setAbsoluteImageDirectory(image_directory,
                                                 line_contents[2])

    steering_angle = float(line_contents[3])
    throttle_value= float(line_contents[4])
    break_value = float(line_contents[5])
    speed_value = float(line_contents[6])

    # read images from path
    center_image = cv2.imread(center_image_path)
    left_image = cv2.imread(left_image_path)
    right_image = cv2.imread(right_image_path)

    return (center_image,
            left_image,
            right_image,
            steering_angle,
            throttle_value,
            break_value,
            speed_value)

# read all data in csv file
with open(log_file_path) as csv_log:
    lines = csv_log.read().splitlines()
    parsedData = [parseLogLine(line) for line in lines]
    parsedSplittedData = [x for x in zip(*parsedData)];

# save parsed data in variable
center_images = parsedSplittedData[0]
left_images = parsedSplittedData[1]
right_images = parsedSplittedData[2]
steering_angles = parsedSplittedData[3]
throttle_values = parsedSplittedData[4]
break_values = parsedSplittedData[5]
speed_values = parsedSplittedData[6]

X_train = np.concatenate((left_images,
                          center_images,
                          right_images),3)

Y_train = np.array(steering_angles)


# create training data
from keras.models import Sequential
from keras.layers import Flatten, Dense

model = Sequential()

model.add(Flatten(input_shape=(160,320,9)))
model.add(Dense(1))

model.compile(loss='mse', optimizer='adam')
model.fit(X_train,Y_train,validation_split=0.2,shuffle=True,nb_epoch=7)

model.save('model.h5')
