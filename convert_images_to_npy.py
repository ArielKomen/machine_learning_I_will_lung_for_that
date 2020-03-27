#code I stole from the internet to convert the input images to .npy arrays. 
import glob
import numpy as np
import cv2

def main():
    #train_pathes = ["/home/cole/Documents/course_10/image_analysis_lessen/image_analysis_eind_opdracht/data/train/effusion_train_images/*.png", "/home/cole/Documents/course_10/image_analysis_lessen/image_analysis_eind_opdracht/data/train/pneumonia_train_images/*.png","/home/cole/Documents/course_10/image_analysis_lessen/image_analysis_eind_opdracht/data/train/pneumonthorax_train_images/*.png"]
    train_pathes = ["/home/cole/Documents/course_10/image_analysis_lessen/image_analysis_eind_opdracht/data/train/effusion_train_images/*.png"]
    test_pathes = ["/home/cole/Documents/course_10/image_analysis_lessen/image_analysis_eind_opdracht/data/test/effusion_validation_data/*.png",
                   "/home/cole/Documents/course_10/image_analysis_lessen/image_analysis_eind_opdracht/data/test/pneumonia_validation_data/*.png"]
    convert_images(train_pathes)
    #convert_images(test_pathes)
    
def convert_images(pathes):
    image_array = []
    image_labels = []
    label = 0
    for path in pathes:
        image_array, image_labels = get_data(path, image_array, image_labels, label)
        label += 1
    np_image_array, np_image_labels = image_to_np_array(image_array, image_labels)
    save_np_array(np_image_array, np_image_labels)

def get_image(path):
    files = glob.glob(path)
    for file in files:
        yield cv2.imread(file)

def get_data(path, image_array, label_array, label):
    for i in range(len(glob.glob(path))):
        image = next(get_image(path))
        image_array.append(image)
    return image_array, label_array.extend([label] * len(image_array))

def image_to_np_array(image_array, label_array):
    image_array = np.array(image_array,dtype='float32') #as mnist
    np_label_array = np.array(label_array,dtype='float64') #as mnist
    # convert (number of images x height x width x number of channels) to (number of images x (height * width *3)) 
    # for example (120 * 40 * 40 * 3)-> (120 * 4800)
    np_image_array = np.reshape(image_array,[image_array.shape[0],image_array.shape[1]*image_array.shape[2]*image_array.shape[3]])
    return np_image_array, np_label_array

def save_np_array(np_image_array, np_label_array):
    # save numpy array as .npy formats
    np.save('train',np_image_array)
    np.save('train_labels',np_label_array)



main()
