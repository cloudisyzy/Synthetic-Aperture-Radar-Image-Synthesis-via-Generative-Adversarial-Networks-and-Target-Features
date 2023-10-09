import os
import random
import shutil
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
import sys
from __models.Classification import get_SAR_dataset, PrincipalComponentsAnalysis, train_classifier, model_evaluation

def compare_arrays(arr1, arr2):
    indices = []

    for i in range(arr1.size):
        if arr1[i] == arr2[i]:
            indices.append(i)

    return indices


def image_filter(filter_dir, destination, train_dir='../SAR_128_train_test/TRAIN/', pca_latent_size=32, crop_size=(64, 64), mode='nb'):
    X_train, y_train = get_SAR_dataset(base_dir=train_dir, crop_size=crop_size, nsamples_subdir_base='all')
    pca = PCA(n_components=pca_latent_size).fit(X_train)
    X_train = pca.transform(X_train)
    model = train_classifier(train_img=X_train, train_label=y_train, mode=mode)

    crop_size = crop_size

    base_dir = filter_dir
    subdirs = os.listdir(base_dir)
    for subdir in subdirs:
        data = []
        labels = []
        subpath = os.path.join(base_dir, subdir)
        if os.path.isdir(subpath):
            files = os.listdir(subpath)
            files = [f for f in files if f.endswith(".jpeg")]
            for file in files:
                filepath = os.path.join(subpath, file)
                image = Image.open(filepath)
                # an image may contain redundant information in the boundary, so crop it
                width, height = image.size
                left = (width - crop_size[0]) / 2
                top = (height - crop_size[1]) / 2
                right = (width + crop_size[0]) / 2
                bottom = (height + crop_size[1]) / 2
                image = image.crop((left, top, right, bottom))
                data.append(np.array(image).flatten())
                labels.append(subdir)

        data = np.array(data) / 255.0
        labels = np.array(labels)
        data = pca.transform(data)
        predict_labels = model.predict(data)
        idx = compare_arrays(predict_labels, labels)
        selected_files = [files[i] for i in idx]
        aim_dir = destination
        if not os.path.exists(aim_dir):
            os.makedirs(aim_dir)
        dst = os.path.join(aim_dir, subdir)
        os.makedirs(dst)
        for file in selected_files:
            src = os.path.join(base_dir, subdir, file)
            shutil.copy(src, dst)