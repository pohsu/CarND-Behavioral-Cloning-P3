import csv
import cv2
import numpy as np
import keras
from keras.models import Sequential, load_model
from keras.layers import Cropping2D
from keras.layers.convolutional import Convolution2D
from keras.layers.core import Dense, Activation, Flatten, Lambda
from keras.layers.pooling import MaxPooling2D
import sklearn
from random import shuffle
from sklearn.model_selection import train_test_split
import copy
import argparse

# argument format: python model.py -f foldername -e 5 -l 5e-4

def main():
    parser = argparse.ArgumentParser(description='Train Keras Model')
    parser.add_argument('-f', '--foldernames', nargs='+', type=str)
    parser.add_argument('-l', '--lr_late', default = 5e-4, type=float)
    parser.add_argument('-e', '--epcohs', default = 5, type=int)
    args = parser.parse_args()
    # provid selected folder names for collecting data
    # data is collected in Windows with the path format:
    #\\data\\foldernames\\IMG\\center_2017_10_26_01_32_48_251.jpg'
    # but traninig is done in linux so '\\' has to be replaced by '/'
    foldernames = args.foldernames
    samples = []
    for foldername in foldernames:
        print('Including data from folder: {}'.format(foldername))
        with open('data/' + foldername + '/driving_log.csv') as csvfile:
            reader = csv.reader(csvfile)
            for line in reader:
                samples.append(line)


    train_samples, validation_samples = train_test_split(samples, test_size=0.2)
    print('Train Sample Size = {}'.format(len(train_samples)))
    print('Valid Sample Size = {}'.format(len(validation_samples)))

    # Define Data Generator with augmentation function inclduing center, right, left images
    # and their flipped ones. When shuffing, I have two different shuffle lists for original and
    # fliiped images so it's better for model generalization
    def generator(samples, batch_size=32):
        num_samples = len(samples)
        while 1: # Loop forever so the generator never terminates
            samples_flipped = copy.copy(samples)
            shuffle(samples)
            shuffle(samples_flipped)
            for offset in range(0, num_samples, batch_size):
                batch_samples = samples[offset:offset+batch_size]
                batch_samples_flipped = samples_flipped[offset:offset+batch_size]
                images = []
                angles = []
                correction_left = 0.2
                correction_right = 0.2
                amp_factor = 1.0
                for batch_sample in batch_samples:
                    name = 'data/' + batch_sample[0].split('\\')[-3] + '/IMG/' + batch_sample[0].split('\\')[-1]
                    center_image = cv2.imread(name)
                    center_angle = float(batch_sample[3])
                    images.append(center_image)
                    angles.append(center_angle)
                    name = 'data/' + batch_sample[1].split('\\')[-3] + '/IMG/' + batch_sample[1].split('\\')[-1]
                    left_image = cv2.imread(name)
                    left_angle = float(batch_sample[3]) + correction_left
                    images.append(left_image)
                    angles.append(left_angle)
                    name = 'data/' + batch_sample[2].split('\\')[-3] + '/IMG/' + batch_sample[2].split('\\')[-1]
                    right_image = cv2.imread(name)
                    right_angle = float(batch_sample[3]) - correction_right
                    images.append(right_image)
                    angles.append(right_angle)

                for batch_sample_flipped in batch_samples_flipped:
                    name = 'data/' + batch_sample_flipped[0].split('\\')[-3]  + '/IMG/' + batch_sample_flipped[0].split('\\')[-1]
                    center_image_flipped = np.fliplr(cv2.imread(name))
                    center_angle_flipped = - (float(batch_sample_flipped[3]))
                    images.append(center_image_flipped)
                    angles.append(center_angle_flipped)
                    name = 'data/' + batch_sample_flipped[1].split('\\')[-3]  + '/IMG/' + batch_sample_flipped[1].split('\\')[-1]
                    left_image_flipped = np.fliplr(cv2.imread(name))
                    left_angle_flipped = - (float(batch_sample_flipped[3]) + correction_left)
                    images.append(left_image_flipped)
                    angles.append(left_angle_flipped)
                    name = 'data/' + batch_sample_flipped[2].split('\\')[-3]  + '/IMG/' + batch_sample_flipped[2].split('\\')[-1]
                    right_image_flipped = np.fliplr(cv2.imread(name))
                    right_angle_flipped = - (float(batch_sample_flipped[3]) - correction_right)
                    images.append(right_image_flipped)
                    angles.append(right_angle_flipped)

                # trim image to only see section with road
                X_train = np.array(images)
                y_train = np.array(angles)*amp_factor
                yield sklearn.utils.shuffle(X_train, y_train)

    # this produces 6 times the size of the orignal data
    aug_factor = 6
    print('Augmented Sample Size = {}'.format(len(samples) * aug_factor))

    # Train and Validation data Generator
    train_generator = generator(train_samples, batch_size=32)
    validation_generator = generator(validation_samples, batch_size=32)


    # Deinfe Convolutional Neural Network in Keras
    # Nvidia version, proven to be effecive
    model = Sequential()
    model.add(Lambda(lambda x:x / 255.0 - 0.5,input_shape=(160,320,3)))
    model.add(Cropping2D(cropping=((50,25), (0,0))))
    model.add(Convolution2D(24, 5, 5, subsample = (2,2), activation='relu'))
    model.add(Convolution2D(36, 5, 5, subsample = (2,2), activation='relu'))
    model.add(Convolution2D(48, 5, 5, subsample = (2,2), activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Convolution2D(64, 3, 3, activation='relu'))
    model.add(Flatten())
    model.add(Dense(100, activation='relu'))
    model.add(Dense(50, activation='relu'))
    model.add(Dense(10, activation='relu'))
    model.add(Dense(1))

    # Here is some code for fine-tuning and layer freezing

    # model = Sequential()
    # del model
    # model = load_model('model.h5')

    # Freeze some layers for fine tuning

    # for layer in model.layers:
    #     layer.trainable = False
    # model.layers[-4].trainable = True
    # model.layers[-3].trainable = True
    # model.layers[-2].trainable = True
    # model.layers[-1].trainable = True

    # Train Model
    adam = keras.optimizers.Adam(lr=args.lr_late, beta_1=0.9, beta_2=0.999, epsilon=1e-08, decay=0.1)
    model.compile(loss = 'mse',optimizer = adam)
    model.fit_generator(train_generator, samples_per_epoch= aug_factor*len(train_samples), validation_data=validation_generator,\
                        nb_val_samples=aug_factor*len(validation_samples), nb_epoch=args.epcohs)

    Save the Model
    model.save('model_new.h5')
    print('model is saved')

if __name__ == '__main__':
    main()
