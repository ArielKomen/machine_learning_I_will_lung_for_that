#image analysis eind opdracht code 
import numpy as np
import pandas as pd
from keras.callbacks import EarlyStopping
from keras.callbacks import TensorBoard
from keras.layers import Dense, Activation, Flatten, Dropout
from keras.layers import MaxPooling2D
from keras.layers.convolutional import Conv2D
from keras.models import Sequential
from keras.utils import np_utils
from keras.utils import multi_gpu_model
from sklearn.metrics import precision_score, recall_score, f1_score
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

def main():
    batch_size = 100
    nb_classes = 15
    nb_epoch = 20
    nb_gpus = 8
    img_rows, img_cols = 512, 512
    channels = 1
    nb_filters = 32
    kernel_size = (2, 2)
   
    labels, disease_X_images = import_data()
    fitted_labels = encode_labels(labels)
    disease_X_train_array, disease_X_test_array, disease_y_train_array, disease_y_test_array = split_data(disease_X_images, fitted_labels, test_data_size)
    
    disease_X_train_reshaped_array, disease_X_test_reshaped_array = reshape_data(disease_X_train_array, disease_X_test_array, img_rows, img_cols, channels)
    input_shape = get_input_shape(img_rows, img_cols, channels)

    disease_X_train_normalized_array, disease_X_test_normalized_array = normalize_data(disease_X_train_reshaped_array, disease_X_test_reshaped_array)
    disease_y_train_matrix, disease_y_test_matrix = transform_categorical_data(disease_y_train_array, disease_y_test_array)

    model = create_model(disease_X_train_normalized_array, disease_y_train_matrix, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes, nb_gpus)




    disease_y_prediction = test_model(model, disease_X_test_normalized_array, disease_y_test_matrix) 
    precision, recall, f1 = calculate_results(disease_y_test_matrix, disease_y_prediction)
    print_results(precision, recall, f1)

def import_data():
    """
    import the data.
    Disease_X_images contains the images for disease X.
    Labels contains per image a label noting what the disease is.
    Returns a numpy array of images containing the disease and the associated labels. 
    """
    disease_X_images = np.load("../data/X_sample.npy")
    labels = pd.read_csv("../data/sample_labels.csv")
    return labels, disease_X_images

def encode_labels(labels):
    """
    Transform the labels(text) into numbers. For example: ["aap", "aap", "vis", "banaan"] = [0, 0, 1, 2]
    input:
         labels: panda series
    output:
        fitted labels, see example above. 
    """
    fitted_labels = labels.Finding_Labels
    # fitted_labels = np.array(pd.get_dummies(fitted_labels))
    label_encoder = LabelEncoder()
    fitted_labels = label_encoder.fit_transform(fitted_labels)
    fitted_labels = fitted_labels.reshape(-1, 1)
    return fitted_labels

def split_data(disease_X_images, fitted_labels, test_data_size=0.2):
    """
    Split data into test and training datasets.
    input:
        disease_X_images: NumPy array of arrays
        fitted_labels   : Pandas series, which are the labels for input array X
        test_data_size  : size of test/train split. Value from 0 to 1 (default=0.2)
    output:
        Four arrays: disease_X_train_array, disease_X_test_array, disease_y_train_array, disease_y_test_array
    """
    print("Splitting data into test/ train datasets")
    disease_X_train_array, disease_X_test_array, disease_y_train_array, disease_y_test_array = train_test_split(disease_X_images, fitted_labels, test_size=test_data_size)
    return disease_X_train_array, disease_X_test_array, disease_y_train_array, disease_y_test_array

def reshape_data(disease_X_train_array, disease_X_test_array, img_rows, img_cols, channels):
    """
    Reshape the data into the format for CNN.
    Input:
         disease_X_train_array: NumPy X_disease train array dataset
         disease_X_test_array : NumPy X_disease test array dataset
         img_rows             : int denoting the amount of favored rows
         img_cols             : int denoting the amount of favored collums
         channels             : Specify if the image is grayscale (1) or RGB (3)
    output:
          disease_X_train_reshaped_array
          disease_X_test_reshaped_array
    """
    print("Reshaping Data")
    disease_X_train_reshaped_array = disease_X_train_array.reshape(disease_X_train_array.shape[0], img_rows, img_cols, channels)
    disease_X_test_reshaped_array = X_test.reshape(X_test.shape[0], img_rows, img_cols, channels)

    print("X_train Shape: ", disease_X_train_reshaped_array.shape)
    print("X_test Shape: ", disease_X_test_reshaped_array.shape)
    return disease_X_train_reshaped_array, disease_X_test_reshaped_array

    
def get_input_shape(img_rows, img_cols, channels):
    """
    get the input shape.
    input:
         img_rows: the amount of rows the images have
         img_cols: the amount of collums the images have
         channels: specify if the image is grayscale (1) or RGB (3)
    output:
         input_shape: a tuple denoting the input shape of all images 
    """
    input_shape = (img_rows, img_cols, channels)
    return input_shape

def normalize_data(disease_X_train_reshaped_array, disease_X_test_reshaped_array):
    """
    Normalize the data.
    input:
         disease_X_train_reshaped_array: NumPy X_disease train array dataset reshaped
         disease_X_test_reshaped_array : NumPy X_disease test array dataset reshaped
    output:
         disease_X_train_normalized_array: NumPy X_disease train array dataset reshaped and normalized
         disease_X_test_normalized_array : NumPy X_disease test array dataset reshaped and normalized
    """
    print("Normalizing Data")
    disease_X_train_reshaped_array = disease_X_train_reshaped_array.astype('float32')
    disease_X_test_reshaped_array = disease_X_test_reshaped_array.astype('float32')

    disease_X_train_normalized_array /= 255
    disease_X_test_normalized_array /= 255

    return disease_X_train_normalized_array, disease_X_test_normalized_array

def transform_categorical_data(disease_y_train_array, disease_y_test_array, nb_classes):
    """
    transform the Y_disease NumPy array dataset into a one-hot-encoding matrix.
    for example: [0,1,0,2] = [[1,0,0][0,1,0][1,0,0][0,0,1]]
    input:
         disease_y_train_array: NumPy y_disease train array dataset
         disease_y_test_array : NumPy y_test train array dataset.
         nb_classes           : total number of classes
    output:
         disease_y_train_matrix: a matrix containing categorical data of y_disease
         disease_y_test_matrix : a matrix containing categorical data of y_disease
    """
    disease_y_train_matrix = np_utils.to_categorical(disease_y_train_array, nb_classes)
    disease_y_test_matrix = np_utils.to_categorical(disease_y_test_array, nb_classes)
    
    print("y_train Shape: ", disease_y_train_matrix.shape)
    print("y_test Shape: ", disease_y_test_matrix.shape)
    return disease_y_train_matrix, disease_y_test_matrix

def create_model(disease_X_train_normalized_array, disease_y_train_matrix, kernel_size, nb_filters, channels, nb_epoch, batch_size, nb_classes, nb_gpus, input_shape):
    """
    create the model existing of 3 layers. Firstly create a model that convolves the images(3 times),
    then apply flatten and dropout layers to prevent overfitting,
    compile the model and lastly train the model using the early predefined layers. 
    input:
        disease_X_train_normalized_array: Array of NumPy arrays
        disease_y_train_matrix: Array of labels
        kernel_size: Initial size of kernel
        nb_filters: Initial number of filters
        channels: Specify if the image is grayscale (1) or RGB (3)
        nb_epoch: Number of epochs
        batch_size: Batch size for the model
        nb_classes: Number of classes for classification
        nb_gpus:  number of GPUs or list of GPU IDs on which to create model replicas
        input_shape: tuple with the input shape of the images
    output:
         model: a fitted CNN model 
    """
    model = get_model()
    model = add_convolving_layers_to_model(model, input_shape, nb_filters, kernel_size)
    model = add_convolving_layers_to_model(model, input_shape, nb_filters=64, kernel_size=(4, 4))
    model = add_convolving_layers_to_model(model, input_shape, nb_filters=128, kernel_size=(8, 8))
    model = flatten_and_add_dropout_layers_to_model(model)
    model = compile_model(model, nb_gpus)
    model = train_model(model, disease_X_train_normalized_array, disease_y_train_matrix, nb_epoch, batch_size)
    return model

def get_model():
    """
    define the model.
    input:
         -
    output:
         model: a initialized Sequential model. 
    """
    model = Sequential()
    return model

def add_convolving_layers_to_model(model, input_shape, nb_filters, kernel_size):
    """
    Add to the model convolving and ReLU layers.
    First set of three layers
    Image size: 256 x 256
    nb_filters = 32
    kernel_size = (2,2)
    input:
         model      : a Sequential model
         nb_filters : total amount of filters. 
         kernel_size: initial size of the kernel
         input_shape: the input shape the images have
    output:
         model: A sequential CNN model.
    """
    
    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1]),
                     padding='valid',
                     strides=1,
                     input_shape=input_shape))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))

    model.add(Conv2D(nb_filters, (kernel_size[0], kernel_size[1])))
    model.add(Activation('relu'))

    model.add(MaxPooling2D(pool_size=(2, 2)))
    return model

def flatten_and_add_dropout_layers_to_model(model):
    """
    Flatten and add dropout layers to the model.
    input:
         model: a sequential CNN model
    output:
         model: a sequential CNN model 
    """
    model.add(Flatten())
    print("Model flattened out to: ", model.output_shape)

    model.add(Dense(4096))
    model.add(Activation('relu'))
    model.add(Dropout(0.2))

    model.add(Dense(4096))
    model.add(Activation("relu"))
    model.add(Dropout(0.2))

    model.add(Dense(nb_classes))
    model.add(Activation('softmax'))
    return model

def compile_model(model, nb_gpus):
    """
    compile the model to use multiple GPU's.
    input:
         model: a sequential CNN model
         nb_gpus:  number of GPUs or list of GPU IDs on which to create model replicas
    output:    
         model: a sequential CNN model
    """
    model = multi_gpu_model(model, gpus=nb_gpus)
    model.compile(loss='categorical_crossentropy',
                  optimizer='adam',
                  metrics=['accuracy'])

    print(model.summary())
    return model

def train_model(model, disease_X_train_normalized_array, disease_y_train_matrix, nb_epoch, batch_size):
    """
    The model is defined using above defined layers. Now it is time to train(update the weights of the filters) the model.
    input:
         model                           : a sequential CNN model
         disease_X_train_normalized_array: NumPy array of normalized X_disease train dataset
         disease_y_train_matrix          : matrix of categorical y_disease data
         nb_epoch                        : Number of epochs
         batch_size                      : batch size for the model 
    output:
         model: a trained sequential CNN model
    """
    stop = EarlyStopping(monitor='acc',
                         min_delta=0.001,
                         patience=2,
                         verbose=0,
                         mode='auto')

    tensor_board = TensorBoard(log_dir='./Graph', histogram_freq=0, write_graph=True, write_images=True)

    model.fit(disease_X_train_normalized_array, disease_y_train_matrix, batch_size=batch_size, epochs=nb_epoch,
              verbose=1,
              validation_split=0.2,
              class_weight='auto',
              callbacks=[stop, tensor_board]
              )
    return model

def test_model(model, disease_X_test_normalized_array, disease_y_test_matrix):
    """
    Test the model.
    input:
         model                          : the fitted CNN model
         disease_X_test_normalized_array: NumPy array with the X_disease test data
         disease_y_test_matrix          : a matrix with the y_disease test data
    output:
         disease_y_prediction: NumPy array with the test/prediction of the model
    """

    print("testing")
    disease_y_prediction = model.predict(disease_X_test_normalized_array)

    disease_y_test_matrix = np.argmax(disease_y_test_matrix, axis=1)
    disease_y_prediction = np.argmax(disease_y_prediction, axis=1)
    return disease_y_prediction


def calculate_results(disease_y_test_matrix, disease_y_prediction):
    """
    calculate the precision, recall and f1 of the disease_y data.
    input:
         disease_y_test_matrix: a matrix with the test data
         disease_y_prediction : NumPy array with the test/prediction of the model
    output:
         precision: the precision of the model
         recall   : the recall of the model
         f1       : the f1 score of the model
    """
    precision = precision_score(y_test, y_pred, average='weighted')
    recall = recall_score(y_test, y_pred, average='weighted')
    f1 = f1_score(y_test, y_pred, average="weighted")
    return precision, recall, f1

def print_results(precision, recall, f1):
    """
    print the results of the model.
    input:
         precision: the precision of the model
         recall   : the recall of the model
         f1       : the f1 score of the model
    """
    print("Precision: ", precision)
    print("Recall: ", recall)
    print("F1: ", f1)



main()
