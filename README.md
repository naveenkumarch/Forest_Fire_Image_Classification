# Forest Fire Image Classification Using Transfer Learning

### Dataset
* The dataset has been Uploaded to Kaggle and the Trained weight files are also uploded to kaggle and made public since Github doesn't support files above 100mb [here](https://www.kaggle.com/chandranaveenkumar/forest-fires-classification)
* Training dataset: This dataset has 39,375 frames that are resized to 254x254 and the image format is JPG.It is in a directory called Training/Training. The data is placed in two folders with respect to class labels.
* Test dataset : Test datset has 8,617 frames that are labeled.This data is in a directory called Test/Test, It also contains 2 sub-directories, one per class.
* The data set also contains two model .h5 files Xception_model and Xception_best. Xception_model was the weight file of the model before applying fine tuning and Xception_best was fine tuned model. (Note: For Evaluation Xception_best model should be used)


### Model
* Transfer Learning based Xception model is used for binary image classification of Fire and Non_fire.
* In the first phase the top layers are removed and xception model works as feature extractor only and in second phase complete Xception model is made trinable except batch normalization layers for fine tuning.
* The Code folder contains Traingin_and_Fine_Tuning jupyter notebook and also python code file which is used for Traing and fine tuning the Xception model.
* The Code folder also contains Evaluation jupyter notebook and python files for loading the best model weights and making predictions on test data.
* Augmentation_script folder containing data-augmentation.py script used for generating new image samples by applying random augmentations. 

## Requirements
* The Traingin_and_Fine_Tuning code is developed for utilizing Kaggle TPU. so it should be executed in kaggle environment and no special environemnt preferences are needed for this code. Where as the evaluation code doesn't Tpu but instead uses Image data generator pipeline form keras. But the model have to be downloded from kaggle dataset.


## Results
* The following are the classification accuracy and Confusion Matrix of the Fine tuned model:
### Classification report
![Classification Report](https://github.com/naveenkumarch/Forest_Fire_Image_Classification/blob/main/Output_pics/New_model_CReport.PNG?raw=true)
### Confusion marix
![Confusion matrix](https://github.com/naveenkumarch/Forest_Fire_Image_Classification/blob/main/Output_pics/Final_result.png?raw=true)
