import os
import random
import numpy as np
from PIL import Image
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.neighbors import KNeighborsClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.metrics import classification_report
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns

def get_SAR_dataset(base_dir, crop_size, nsamples_subdir_base='all', nsamples_subdir_extended=512, extended_dir=None):
#    This function intends to return dataset and labels that are usable in sklearn 
    data = []
    labels = []
    
    # create the datasets of base dir
    subdirs = os.listdir(base_dir)
    for subdir in subdirs:
        subpath = os.path.join(base_dir, subdir)
        if os.path.isdir(subpath):
            files = os.listdir(subpath)
            files = [f for f in files if f.endswith(".jpeg")]
            # choose how many images to be included in each sub-class, 'all' indicates returning all of them
            if nsamples_subdir_base == 'all':
                selected_files = files
            else:
                selected_files = random.sample(files, nsamples_subdir_base)
            for file in selected_files:
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
                
    # create the datasets of extended dir and concatenate it with base dir, only useful when create train data combination of original images and generated samples)
    if extended_dir != None:
        subdirs = os.listdir(extended_dir)
        for subdir in subdirs:
            subpath = os.path.join(extended_dir, subdir)
            if os.path.isdir(subpath):
                files = os.listdir(subpath)
                files = [f for f in files if f.endswith(".jpeg")]
                if nsamples_subdir_extended > len(files):
                    selected_files = files
                else:
                    selected_files = random.sample(files, nsamples_subdir_extended)
                for file in selected_files:
                    filepath = os.path.join(subpath, file)
                    image = Image.open(filepath)
                    width, height = image.size
                    left = (width - crop_size[0]) / 2
                    top = (height - crop_size[1]) / 2
                    right = (width + crop_size[0]) / 2
                    bottom = (height + crop_size[1]) / 2
                    image = image.crop((left, top, right, bottom))
                    data.append(np.array(image).flatten())
                    labels.append(subdir)
                
    X = np.array(data) / 255.0
    y = np.array(labels)
    
    return X, y


def PrincipalComponentsAnalysis(pca_latent_size, trainset, testset=None):
#    This function fits Principal-Components-Analysis to trainset and apply it to testset, then return the modified dataset
    pca = PCA(n_components=pca_latent_size).fit(trainset)
    print('Length of Train Dataset: %d' %len(trainset))
    if testset.all() != None:
        print('Length of Test Dataset: %d\n\n' %len(testset))
        return pca.transform(trainset), pca.transform(testset)
    else:
        return pca.transform(trainset)

    
def train_classifier(train_img, train_label, mode='svm'):
    # User will choose a type of classifier ('svm' is recommended) out of three and train it using trainset, the function will return the trained model
    if mode == 'svm':
        # Radial Basis Function Support Vector Machine
        model = SVC(C=1, kernel='rbf', max_iter=-1, random_state=0)
    if mode == 'knn':
        # 10-Nearest Neighbors Classifier
        model = KNeighborsClassifier(n_neighbors=10, weights="uniform", algorithm="auto")
    if mode == 'nb':
        # Gaussian Naive Bayes Classifier
        model = GaussianNB()
        
    model.fit(train_img, train_label)
    print('Training Accuracy is: %.4f%%\n\n' % (model.score(train_img, train_label)*100))
    return model


def model_evaluation(model, test_img, test_label):
#    This function aims to evaluate the performance of the classifier over the test data
    # First, print the classification accuracy
    # Second, print the classification report
    # Third, print the confusion matrix
    predictions = model.predict(test_img)
    report = classification_report(test_label, predictions)
    matrix = confusion_matrix(test_label, predictions)
    accuracy = model.score(test_img, test_label)*100
    print('Test Accuracy is: %.4f%%\n\n' % accuracy)
    print(report, '\n\n')
    plt.rcParams['figure.dpi'] = 200
    class_labels = ['2S1','BMP2','BRDM2','BTR60','BTR70','D7','T62','T72','ZIL131','ZSU234']
    sns.heatmap(matrix, annot=True, fmt="d", cmap="icefire", xticklabels=class_labels, yticklabels=class_labels)
    plt.show()
    return accuracy