#!/usr/bin/env python
# coding: utf-8

# In[ ]:


import sys
sys.path.append("..")
from __models.Classification import get_SAR_dataset, PrincipalComponentsAnalysis, train_classifier, model_evaluation
sys.path.remove("..")


# In[ ]:


train_dir = '../_MSTAR/TRAIN/'
extended_train_dir = '../_MSTAR/TRAIN/'
num_extended_per_class = 0
test_dir = '../_MSTAR/TEST/'


# In[ ]:


X_train, y_train = get_SAR_dataset(base_dir=train_dir, crop_size=(64, 64), nsamples_subdir_base='all', 
                                   extended_dir=extended_train_dir, nsamples_subdir_extended=num_extended_per_class)

X_test, y_test = get_SAR_dataset(base_dir=test_dir, crop_size=(64, 64), nsamples_subdir_base='all')

X_train, X_test = PrincipalComponentsAnalysis(pca_latent_size=16, trainset=X_train, testset=X_test)

model = train_classifier(train_img=X_train, train_label=y_train, mode='svm')

accuracy = model_evaluation(model=model, test_img=X_test, test_label=y_test)

