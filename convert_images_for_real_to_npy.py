#this is code I stole from the internet to convert images to a .npy file. 
import cv2
import os
import time
import numpy as np
import pandas as pd

def main():
    
    #crop_and_resize_initial_images()
    #create_csv()
    start_time = time.time()

    labels = pd.read_csv("data/sample_labels.csv")

    print("Writing Train Array")
    X_train = convert_images_to_arrays('data/resized-256/', labels)

    print(X_train.shape)

    print("Saving Train Array")
    save_to_array('data/X_sample.npy', X_train)

    print("Seconds: ", round(time.time() - start_time), 2)

def crop_and_resize_initial_images():
    start_time = time.time()
    crop_and_resize_images(path='data/images/', new_path='data/resized-256/', img_size=256)
    print("Seconds: ", time.time() - start_time)
    
def create_csv():
    #create a csv of the labels information using panda's. 
    data = pd.read_csv("data/Data_Entry_2017.csv")
    sample = os.listdir('data/resized-256/')

    sample = pd.DataFrame({'Image Index': sample})

    sample = pd.merge(sample, data, how='left', on='Image Index')

    sample.columns = ['Image_Index', 'Finding_Labels', 'Follow_Up_#', 'Patient_ID',
                      'Patient_Age', 'Patient_Gender', 'View_Position',
                      'Original_Image_Width', 'Original_Image_Height',
                      'Original_Image_Pixel_Spacing_X',
                      'Original_Image_Pixel_Spacing_Y', 'Unnamed']

    sample['Finding_Labels'] = sample['Finding_Labels'].apply(lambda x: x.split('|')[0])

    sample.drop(['Original_Image_Pixel_Spacing_X', 'Original_Image_Pixel_Spacing_Y', 'Unnamed'], axis=1, inplace=True)
    sample.drop(['Original_Image_Width', 'Original_Image_Height'], axis=1, inplace=True)

    print("Writing CSV")
    sample.to_csv('data/sample_labels.csv', index=False, header=True)

def create_directory(directory):
    """
    Creates a new folder in the specified directory if folder doesn't exist.
    INPUT
        directory: Folder to be created, called as "folder/".
    OUTPUT
        New folder in the current directory.
    """
    if not os.path.exists(directory):
        os.makedirs(directory)


def crop_and_resize_images(path, new_path, img_size):
    """
    Crops, resizes, and stores all images from a directory in a new directory.
    INPUT
        path: Path where the current, unscaled images are contained.
        new_path: Path to save the resized images.
        img_size: New size for the rescaled images.
    OUTPUT
        All images cropped, resized, and saved to the new folder.
    """
    create_directory(new_path)
    dirs = [l for l in os.listdir(path) if l != '.DS_Store']
    # total = 0

    for item in dirs:
        # Read in all images as grayscale
        img = cv2.imread(path + item, cv2.IMREAD_GRAYSCALE)
        img = cv2.resize(img, (img_size, img_size))
        cv2.imwrite(str(new_path + item), img)
        # total += 1
        # print("Saving: ", item, total)

def get_lst_images(file_path):
    """
    Reads in all files from file path into a list.
    INPUT
        file_path: specified file path containing the images.
    OUTPUT
        List of image strings
    """
    return [i for i in os.listdir(file_path) if i != '.DS_Store']


def convert_images_to_arrays(file_path, df):
    """
    Converts each image to an array, and appends each array to a new NumPy
    array, based on the image column equaling the image file name.
    INPUT
        file_path: Specified file path for resized test and train images.
        df: Pandas DataFrame being used to assist file imports.
    OUTPUT
        NumPy array of image arrays.
    """

    lst_imgs = [l for l in df['Image_Index']]

    return np.array([np.array(cv2.imread(file_path + img, cv2.IMREAD_GRAYSCALE)) for img in lst_imgs])


def save_to_array(arr_name, arr_object):
    """
    Saves data object as a NumPy file. Used for saving train and test arrays.
    INPUT
        arr_name: The name of the file you want to save.
            This input takes a directory string.
        arr_object: NumPy array of arrays. This object is saved as a NumPy file.
    """
    return np.save(arr_name, arr_object)
    

main()
