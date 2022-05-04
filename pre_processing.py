from keras.preprocessing.image import img_to_array, load_img
import numpy as np
import os
import glob
from PIL import Image

#load images from directory, convert to grayscale, add to numpy array, return array
def image_to_array(files, destination, size, BW=False):
    i=0
    for filepath in files:
        i += 1
        image = Image.open(filepath)
        image = image.resize((size, size)) #resize image to 28x28
        if(BW):
            image = image.convert("L") #converts image to Grayscale
        image.save(destination+str(i)+'.jpg', 'JPEG')


train_files = glob.glob('train/*.jpg')
image_to_array(train_files,'x_train/', 256)

test_files = glob.glob('train/*.jpg')
image_to_array(test_files, 'y_test/', 256, True)

# Get images
X = []
for filename in os.listdir('x_train/'):
    X.append(img_to_array(load_img('x_train/'+filename)))
X = np.array(X, dtype=float)

#save numpy arrays to file
np.save('cat_array.npy', X)