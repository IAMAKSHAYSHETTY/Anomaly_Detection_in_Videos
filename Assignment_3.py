# Importing all the necessary libraries
import os
import cv2
import keras
from keras import layers
from keras.models import load_model
import tensorflow as tf
from PIL import Image
from glob import glob
import numpy as np
from matplotlib import pyplot as plt


# Relative paths- Change the paths as necessary

frame_path = 'frame0731.jpg' #Change the path to the path of the frame to test
model_path = 'AutoEncoder_Model' #Path of the trained model

# Converting Video to images frame by frame

#Converting Video to images frame by frame
def convert_video_to_images(img_folder, filename='assignment3_video.avi'):
  # Make the img_folder if it doesn't exist.'
  try:
    if not os.path.exists(img_folder):
      os.makedirs(img_folder)
  except OSError:
    print('Error')
  # Make sure that the abscense/prescence of path
  # separator doesn't throw an error.
  img_folder = f'{img_folder.rstrip(os.path.sep)}{os.path.sep}'
  # Instantiate the video object.
  video = cv2.VideoCapture(filename)
  # Check if the video is opened successfully
  if not video.isOpened():
    print("Error opening video file")
  i = 0
  while video.isOpened():
    ret, frame = video.read()
    if ret:
      im_fname = f'{img_folder}frame{i:0>4}.jpg'
      print('Captured...', im_fname)
      cv2.imwrite(im_fname, frame)
      i += 1
    else:
      break
  video.release()
  cv2.destroyAllWindows()
  if i:
    print(f'Video converted\n{i} images written to {img_folder}')

convert_video_to_images('image_data')

# Flattening and resizing the images in the directory


def load_images(img_dir, im_width=60, im_height=44):
    images = []
    fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
    fnames.sort()
    for fname in fnames:
        im = Image.open(fname)
        # resize the image to im_width and im_height.
        im_array = np.array(im.resize((im_width, im_height)))
        # Convert uint8 to decimal and normalize to 0 - 1.
        images.append(im_array.astype(np.float32) / 255.)
        # Close the PIL image once converted and stored.
        im.close()
        # Flatten the images to a single vector
    X = np.array(images).reshape(-1, np.prod(images[0].shape))
    return X, images


# Calling the flattening function
X_train, images = load_images('image_data')

# Building the Autoencoder Structure
input = tf.keras.layers.Input(shape=(X_train.shape[1],))
# Encoder layers
encoder = tf.keras.Sequential([
    layers.Dense(16, activation='relu'),
    layers.Dense(8, activation='relu'),
    layers.Dense(4, activation='relu')])(input)
# Decoder layers
decoder = tf.keras.Sequential([
    layers.Dense(8, activation="relu"),
    layers.Dense(16, activation="relu"),
    layers.Dense(7920, activation="sigmoid")])(encoder)
# Create the autoencoder
autoencoder = tf.keras.Model(input, decoder)

autoencoder.compile(optimizer='adam', loss='binary_crossentropy')
# Fit the autoencoder
autoencoder.fit(X_train, X_train,
                epochs=50,
                batch_size=128,
                shuffle=True)

autoencoder.save('AutoEncoder_Model')

# Finiding Threshold
# finding loss before the Anomaly
'''images = []
im_width = 60
im_height = 44
loss_before = []
img_dir = 'Before_Canoe'
fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
fnames.sort()
for fname in fnames:
    im = Image.open(fname)
    # resize the image to im_width and im_height.
    im_array = np.array(im.resize((im_width, im_height)))
    # Convert uint8 to decimal and normalize to 0 - 1.
    images.append(im_array.astype(np.float32) / 255.)
    # Close the PIL image once converted and stored.
    im.close()
    Xa = np.array(images).reshape(-1, np.prod(images[0].shape))
    loss_before.append(autoencoder.evaluate(Xa, Xa, verbose=0))

# finding loss During the Anomaly
images = []
loss_during = []
img_dir = 'Canoe'
fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
fnames.sort()
for fname in fnames:
    im = Image.open(fname)
    # resize the image to im_width and im_height.
    im_array = np.array(im.resize((im_width, im_height)))
    # Convert uint8 to decimal and normalize to 0 - 1.
    images.append(im_array.astype(np.float32) / 255.)
    # Close the PIL image once converted and stored.
    im.close()
    Xa = np.array(images).reshape(-1, np.prod(images[0].shape))
    loss_during.append(autoencoder.evaluate(Xa, Xa, verbose=0))

# finding loss after the Anomaly
images = []
loss_after = []
img_dir = 'After_Canoe'
fnames = glob(f'{img_dir}{os.path.sep}frame*.jpg')
fnames.sort()
for fname in fnames:
    im = Image.open(fname)
    # resize the image to im_width and im_height.
    im_array = np.array(im.resize((im_width, im_height)))
    # Convert uint8 to decimal and normalize to 0 - 1.
    images.append(im_array.astype(np.float32) / 255.)
    # Close the PIL image once converted and stored.
    im.close()
    Xa = np.array(images).reshape(-1, np.prod(images[0].shape))
    loss_after.append(autoencoder.evaluate(Xa, Xa, verbose=0))

# Joining all the losses
total_loss = loss_before + loss_during + loss_after

# Plotting the loss
plt.plot(total_loss)'''

# function to predict if the frame has anomaly or not


def predict(frame):
    model = load_model(model_path)
    images = []
    im_width = 60
    im_height = 44
    im = Image.open(frame)
    # resize the image to im_width and im_height.
    im_array = np.array(im.resize((im_width, im_height)))
    # Convert uint8 to decimal and normalize to 0 - 1.
    images.append(im_array.astype(np.float32) / 255.)
    # Close the PIL image once converted and stored.
    im.close()
    # Flatten the images to a single vector
    Xr = np.array(images).reshape(-1, np.prod(images[0].shape))
    loss_test = model.evaluate(Xr, Xr, verbose=0)
    threshold = 0.515
    anomaly = (False if loss_test < threshold else True)
    return anomaly


print(" Is the Frame Anomalous:", predict(frame_path))
