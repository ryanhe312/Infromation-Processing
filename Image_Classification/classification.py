# %%
## get descriptor

import os
import numpy as np

mpeg7_path = '.\mpeg7fex_win32_v2\MPEG7Fex.exe'

type_dict = {
    'CSD':'128',
    'SCD':'128'
}

def get_mpeg7_descriptor(imagePath, featureType='CSD', featureParameters='64'):
    imageListFile = 'input.txt'
    outputFile = 'output.txt'

    with open(imageListFile,'w') as f:
        f.write(imagePath)
    stream = os.popen(f'{mpeg7_path} {featureType} {featureParameters} {imageListFile} {outputFile}')
    stream.read()
    with open(outputFile,'r') as f:
        lines = np.array([line.strip().split() for line in f.readlines()])
    # print(lines)
    return lines[0,1:].astype(int)


# %%
## load data

train_num = 2000
test_num = 200

from pathlib import Path
dataset_path = Path('..\VOCtrainval_11-May-2012\VOCdevkit\VOC2012')
label_path = os.path.join(dataset_path,'ImageSets','Main')
img_path = os.path.join(dataset_path,'JPEGImages')

def read_label(filePath):
    with open(filePath,'r') as f:
        lines = np.array([line.strip().split() for line in f.readlines()])
    return dict(zip(lines[:,0],lines[:,1].astype(int)))

cat_train = read_label(os.path.join(label_path,'cat_train.txt'))
dog_train = read_label(os.path.join(label_path,'dog_train.txt'))
cat_val = read_label(os.path.join(label_path,'cat_val.txt'))
dog_val = read_label(os.path.join(label_path,'dog_val.txt'))

train_list = list(set(cat_train.keys())|set(dog_train.keys()))
test_list = list(set(cat_val.keys())|set(dog_val.keys()))

train_label = []
for img in train_list:
    if img in cat_train.keys() and cat_train[img] == 1:
        train_label.append(1)
    elif img in dog_train.keys() and dog_train[img] == 1:
        train_label.append(2)
    else:
        train_label.append(0)
train_label = np.array(train_label)
    
test_label = []
for img in test_list:
    if img in cat_val.keys() and cat_val[img] == 1:
        test_label.append(1)
    elif img in dog_val.keys() and dog_val[img] == 1:
        test_label.append(2)
    else:
        test_label.append(0)
test_label = np.array(test_label)

train_list = np.concatenate([np.array(train_list)[train_label==1],
                            np.array(train_list)[train_label==2],
                            np.array(train_list)[train_label==0][:500]])
train_label = np.sum(train_label==1) * [1] + np.sum(train_label==2) * [2] + 500 * [0]

test_list = np.concatenate([np.array(test_list)[test_label==1],
                            np.array(test_list)[test_label==2],
                            np.array(test_list)[test_label==0][:500]])
test_label = np.sum(test_label==1) * [1] + np.sum(test_label==2) * [2] + 500 * [0]

test_paths = [os.path.join(img_path,path+'.jpg') for path in test_list]
train_paths = [os.path.join(img_path,path+'.jpg') for path in train_list]
    
print(len(train_list),len(test_list),len(train_label),len(test_label))


# %%
## test 

from tqdm import tqdm
from sklearn import svm
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import accuracy_score,f1_score

model_dict= {
    'svm':svm.SVC,
    'tree':DecisionTreeClassifier
}

for kt,vt in type_dict.items():
    for km,vm in model_dict.items():

        train_features = []
        for path in tqdm(train_paths):
            train_feature = get_mpeg7_descriptor(path,featureType=kt,featureParameters=vt)
            train_features.append(train_feature)
            tqdm._instances.clear()

        train_features = np.array(train_features)
        # print(train_features.shape)

        test_features = []
        for path in tqdm(test_paths):
            test_feature = get_mpeg7_descriptor(path,featureType=kt,featureParameters=vt)
            test_features.append(test_feature)
            tqdm._instances.clear()

        test_features = np.array(test_features)
        # print(test_features.shape)

        model = vm()
        model.fit(train_features,train_label)

        y_pred = model.predict(test_features)
        print(f"The {km} model with {kt} feature accuracy is:",accuracy_score(y_pred,test_label),
            'f1_score is:',f1_score(y_pred,test_label,average='micro'))


