import torch.utils.data as data
import cv2
import pandas as pd
import os
import image_utils
import random



class RafDataSet(data.Dataset):
    def __init__(self, raf_path, phase, transform=None, basic_aug=False, attribute=None):
        self.phase = phase
        self.transform = transform
        self.raf_path = raf_path
        self.attribute = attribute

        NAME_COLUMN = 0
        LABEL_COLUMN = 1
        df = pd.read_csv(os.path.join(self.raf_path, 'EmoLabel/list_patition_label.txt'), sep=' ', header=None)
        if phase == 'train':
            dataset = df[df[NAME_COLUMN].str.startswith('train')]
        else:
            dataset = df[df[NAME_COLUMN].str.startswith('test')]
        file_names = dataset.iloc[:, NAME_COLUMN].values
        self.label = dataset.iloc[:,
                     LABEL_COLUMN].values - 1  # 0:Surprise, 1:Fear, 2:Disgust, 3:Happiness, 4:Sadness, 5:Anger, 6:Neutral

        if self.attribute == 'age':
            df = pd.read_csv(os.path.join(self.raf_path, 'AgeLabel/list_patition_label.txt'), sep='\t', header=None)
            if phase == 'train':
                dataset = df[df[NAME_COLUMN].str.startswith('train')]
            else:
                dataset = df[df[NAME_COLUMN].str.startswith('test')]
            file_names = dataset.iloc[:, NAME_COLUMN].values
            self.age = dataset.iloc[:,LABEL_COLUMN].values 
        elif self.attribute == 'gender':
            df = pd.read_csv(os.path.join(self.raf_path, 'GenderLabel/list_patition_label.txt'), sep='\t', header=None)
            if phase == 'train':
                dataset = df[df[NAME_COLUMN].str.startswith('train')]
            else:
                dataset = df[df[NAME_COLUMN].str.startswith('test')]
            file_names = dataset.iloc[dataset.iloc[:, LABEL_COLUMN].values != 2, NAME_COLUMN].values
            self.gender = dataset.iloc[dataset.iloc[:, LABEL_COLUMN].values != 2,LABEL_COLUMN].values 
        self.file_paths = []
        # use raf-db aligned images for training/testing
        for f in file_names:
            f = f.split(".")[0]
            f = f + "_aligned.jpg"
            path = os.path.join(self.raf_path, 'Image/aligned', f)
            self.file_paths.append(path)

        self.basic_aug = basic_aug
        self.aug_func = [image_utils.flip_image, image_utils.add_gaussian_noise]

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, idx):
        path = self.file_paths[idx]
        image = cv2.imread(path)
        image = image[:, :, ::-1]  # BGR to RGB
        label = self.label[idx]
        if self.phase == 'train':
            if self.basic_aug and random.uniform(0, 1) > 0.5:
                index = random.randint(0, 1)
                image = self.aug_func[index](image)

        if self.transform is not None:
            image = self.transform(image)
        if self.attribute == 'age':
            sensitiveLabel = self.age[idx]
        elif self.attribute == 'gender':
            sensitiveLabel = self.gender[idx]


        return image, label, idx, sensitiveLabel
