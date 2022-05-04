from keras.models import load_model
from keras.preprocessing.image import img_to_array, load_img
from skimage.color import rgb2lab, lab2rgb
from skimage.io import imsave
import numpy as np
import os

#load numpy arrays from file
X = np.load('cat_array.npy',allow_pickle=True)

# Set up train and test data
split = int(0.95*len(X))
Xtrain = X[:split]
Xtrain = 1.0/255*Xtrain
batch_size=20

#load model
model = load_model('my_model')

#evaluate model
Xtest = rgb2lab(1.0/255*X[split:])[:,:,:,0]
Xtest = Xtest.reshape(Xtest.shape+(1,))
Ytest = rgb2lab(1.0/255*X[split:])[:,:,:,1:]
Ytest = Ytest / 128
print("Eval Score: ", model.evaluate(Xtest, Ytest, batch_size=batch_size))

#format test data
color_me = []
for filename in os.listdir('y_test/'):
    color_me.append(img_to_array(load_img('y_test/'+filename)))
color_me = np.array(color_me, dtype=float)
color_me = rgb2lab(1.0/255*color_me)[:,:,:,0]
color_me = color_me.reshape(color_me.shape+(1,))

# Test model
output = model.predict(color_me)
output = output * 128

# Output colorized images
for i in range(len(output)):
    cur = np.zeros((256, 256, 3))
    cur[:,:,0] = color_me[i][:,:,0]
    cur[:,:,1:] = output[i]
    imsave("image_output/img_"+str(i)+".png", (lab2rgb(cur)*255).astype(np.uint8))