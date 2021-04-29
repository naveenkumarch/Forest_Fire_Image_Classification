#!/usr/bin/env python
# coding: utf-8

# Import Libraries
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import numpy as np
from sklearn.metrics import confusion_matrix, accuracy_score,classification_report
import seaborn as sns
import matplotlib.pyplot as plt
from tensorflow import keras 

# Define path to test data 
test_path = "../input/forest-fires-classification/Test/Test"

# Initialize tess data pipeline
test_datagen = ImageDataGenerator(rescale=1./255)
test_generator = test_datagen.flow_from_directory(
        test_path,
        target_size=(254, 254),
        shuffle = False,
        class_mode='binary',
        batch_size=1)


#Loading the modle from weight files
model = keras.models.load_model('../input/forest-fires-classification/Xception_best.h5')


# Extracting True Labels
true_labels = test_generator.classes

# Making Predictions on Test Data
predictions = model.predict_generator(test_generator, steps=len(true_labels))


# Converting sigmoid function outputs to zeros and ones
pred_list = list(predictions)
pred_labels = [1 if entry > 0.5 else 0 for entry in pred_list]

# Calculating the accuracy scores
print("Best Accyracy attained by model is :",accuracy_score(true_labels,pred_labels))


# calculating confusion matrix and plotting heat map of confusion matrix
confu_matrix=confusion_matrix(true_labels,pred_labels)
ax = sns.heatmap(confu_matrix,annot=True,cmap='BuGn', fmt='g',xticklabels=['Fire', 'No_Fire'], yticklabels=['Fire', 'No_Fire']).set_title('Confusion Matrix')
fig = ax.get_figure()
fig.savefig("Final_result.png")


print(classification_report(true_labels,pred_labels))





