import os
import pickle
import numpy as np
from skimage.io import imread
from skimage.transform import resize
from sklearn.model_selection import train_test_split
from sklearn.model_selection import GridSearchCV
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
from joblib import load


# load data
input_dir = 'D:/PyCharm/imageClassificator/clf-data/'
categories = ['car', 'not_car']

data = [] #список для хранения признаков изображений
labels = [] #список для хранения меток классов соответствующих изображений

for category_idx, category in enumerate(categories):
    for file in os.listdir(os.path.join(input_dir, category)):
        img_path = os.path.join(input_dir, category, file)
        img = imread(img_path)
        img = resize(img, (15, 15))
        #size = img.shape
        data.append(img.flatten())
        labels.append(category_idx)

data = np.asarray(data)
labels = np.asarray(labels)

# train / test split
X_train, X_valid, y_train, y_valid = train_test_split(data, labels, test_size=0.2, shuffle=True, stratify=labels)

# train classificator
classifier = SVC()
params = [{'gamma': [0.01, 0.001, 0.0001], 'C': [1, 10, 100, 1000]}]
grid_search = GridSearchCV(classifier, params)
grid_search.fit(X_train, y_train)

# test check
best_estimator = grid_search.best_estimator_
predictions = best_estimator.predict(X_valid)

score = accuracy_score(predictions, y_valid)

print(f'{score*100}% объектов классифицировано верно')

#pickle.dump(best_estimator, open('./model.p', 'wb'))
