# 🌼 Flower Image Classifier

A simple Flask web app that classifies flower images using a trained CNN model in TensorFlow/Keras.

## 🚀 Features

- Upload a flower image and get an instant prediction
- Supports five flower types: daisy, dandelion, rose, sunflower, tulip
- Built with Flask, TensorFlow, and Keras
- Clean and responsive user interface

## 🧠 Model

The model is a Convolutional Neural Network (CNN) trained on a dataset of flower images.  
Trained with:
- `Conv2D`, `MaxPooling`, `Dropout`, `Dense`
- `image_dataset_from_directory` for loading
- `RandomZoom`, `RandomRotation`, and `RandomFlip` for augmentation

## 🛠️ Tech Stack

- Python
- Flask
- TensorFlow / Keras
- HTML + CSS (Jinja2 templating)

## 📦 Dependencies

- Flask
- tensorflow
- keras
- numpy

## 🧪 Dataset and Visualization Checks

- During development, it's helpful to verify your dataset and augmentation visually. 
  Below are optional code snippets you can uncomment and run to explore your training data.
  You can add them in FlowerRecogModel.py:

### 📊 Confirm Dataset Sample Counts
- You can print how many images are in each class to verify the dataset was loaded correctly
- This could be added at line 11 to confirm training dataset samples:

# count = 0 
# dirs = os.listdir('FlowerTrainingDataset/')
# for dir in dirs:
#     files = list(os.listdir('FlowerTrainingDataset/'+dir))
#     print( dir +' Folder has '+ str(len(files)) + ' Images')
#     count = count + len(files)
# print( 'FlowerTrainingDataset Folder has '+ str(count) + ' Images')


### 🖼️ Preview Sample Images
- To visually confirm that the images and labels are loading correctly from the training set:
- This could be added at line 53 to check train_ds:

# i = 0
# plt.figure(figsize=(10,10))

# for images, labels in train_ds.take(1):
#     for i in range(9):
#         plt.subplot(3,3, i+1)
#         plt.imshow(images[i].numpy().astype('uint8'))
#         plt.title(flower_names[labels[i]])
#         plt.axis('off')

# plt.show()

### 🎨 Preview Data Augmentation
- To see how your data augmentation pipeline is transforming images in real-time
- This could be addded at line 68 to check train_ds with filter applied:
# i = 0
# plt.figure(figsize=(10,10))

# for images, labels in train_ds.take(1):
#     for i in range(9):
#         images = data_augmentation(images)
#         plt.subplot(3,3, i+1)
#         plt.imshow(images[0].numpy().astype('uint8'))
#         plt.axis('off')

# plt.show()