# The following code needs to be made in R:
# The X_sample.npy and sample_labels.csv need to be imported. The sample_labels need to be converted to numbers.
# The sample_labels need to converted to a one-hotted encoding matrix.
# The X_sample data needs to be split into test and train datasets. The X_sample data needs to be reshaped,
# into a shape we desire. The X_sample needs to be normalized.
# Instantieer a Sequential model and add 3* a [(Conv+ReLU)*3+MaxPool] layer of which the kernel_size and
# nb_filters continuously 2* becomes bigger(dus: 2,4,8). Afterwards flatten the model and add some dropout layers.
#Then the model is finished to trainen and to test. We would like to know the precision, recall and F1 score.

# "/home/cole/Documents/course_10/image_analysis_lessen/image_analysis_blok_10_eind_opdracht/data/X_sample.npy.gz"
library(reticulate)
#setup python and use image analysis code
python_path <- "/usr/bin/python3"
use_python(python_path, required=TRUE)
#py_config() #show configuration of python 
source_python("/home/cole/Documents/course_10/image_analysis_lessen/image_analysis_blok_10_eind_opdracht/image_analysis_only_functions.py")
#import python packages
np <-import("numpy", convert=FALSE)
pd <-import("pandas")
sklearn <- import("sklearn", convert = FALSE)
keras <- import("keras", convert=FALSE)

#define globals...bad pratice but it works
test_data_size<-0.2
img_rows<-as.integer(256)
img_cols<-as.integer(256)
channels<-as.integer(1)
nb_filters<-as.integer(32)
nb_classes<-as.integer(15)
nb_epoch<-as.integer(20)
batch_size<-as.integer(100)
kernel_size<-tuple(as.integer(2),as.integer(2))

#load in data
disease_X_images <-np$load("/home/cole/Documents/course_10/image_analysis_lessen/image_analysis_blok_10_eind_opdracht/data/X_sample.npy")
labels <-pd$read_csv("/home/cole/Documents/course_10/image_analysis_lessen/image_analysis_blok_10_eind_opdracht/data/sample_labels.csv")
#transform the labels into numbers, for example: ["aap","aap","vis"] = [0,0,1]
fitted_labels <- encode_labels(labels)
#disease_X_train_array; disease_X_test_array; disease_y_train_array; disease_y_test_array
#split the amount of images. The data is split based on the test_data_size global
throw_away_split_variable<-sklearn$model_selection$train_test_split(disease_X_images, fitted_labels, test_size=test_data_size)
disease_X_train_array<-throw_away_split_variable[0]
disease_X_test_array<-throw_away_split_variable[1]
disease_y_train_array<-throw_away_split_variable[2]
disease_y_test_array<-throw_away_split_variable[3]

#reshape the data based on the img_rows and img_cols globals
#!does not work right now, might not be needed at all though.
#np$reshape(np$shape(),img_rows,img_cols,channels)
#throw_away_reshape_variable<-reshape_data(disease_X_train_array, disease_X_test_array, img_rows, img_cols, channels)

#get input shape of the data, this does work
input_shape<-get_input_shape(img_rows,img_cols,channels)

#normalize the data, does not work yet
#throw_away_normalised_variable<-normalize_data(disease_X_train_array, disease_X_test_array)
disease_X_train_array <- np$true_divide(np$float32(disease_X_train_array), 255)
disease_X_test_array <- np$true_divide(np$float32(disease_X_test_array), 255)

#transform the disease_y_test/train_array to a one-hot encoded matrix.
#disease_y_matrix is divided in two parts, the first is the train matrix and the second the test matrix.
disease_y_matrix <- transform_categorical_data(disease_y_train_array, disease_y_test_array, nb_classes)

#use python code to build the model. The python code only accepts integers, so most of the modifications
#are to make sure that a integer is passed to python. 
model <- get_model()
model <- add_convolving_layers_to_model(model, tuple(input_shape), nb_filters, tuple(kernel_size))
model <- add_convolving_layers_to_model(model, tuple(input_shape), nb_filters*as.integer(2), tuple(as.integer(4),as.integer(4)))
model <- add_convolving_layers_to_model(model, tuple(input_shape), nb_filters*as.integer(4), tuple(as.integer(8),as.integer(8)))
model <- flatten_and_add_dropout_layers_to_model(model, nb_classes)
model <- compile_model(model)
model <- train_model(model, disease_X_train_array, disease_y_matrix[1], nb_epoch, batch_size)
