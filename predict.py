import pickle
from skimage.io import imread
from skimage.transform import resize

predsClasses = {0: "car", 1: "not car"}

#preprocessing dataset
model = pickle.load(open('./model.p', 'rb'))
img = imread('image9.jpg')
img = resize(img, (15, 15))
dataset = img.flatten().reshape(1, -1)

prediction = model.predict(dataset)
print(f'Prediction class: {predsClasses[int(*prediction)]}')