import os
import torchvision
from PIL import Image


class ImageDataset(Dataset):
    """Dataset class used for Image Classification.
    """
    def __init__(self, file_list):
        self.file_list = file_list

    def __len__(self):
        return len(self.file_list)

    def __getitem__(self, index):
        img = Image.open(self.file_list[index])
        img = torchvision.transforms.ToTensor()(img)
        return img


class MyDataset(Dataset):
    """Dataset class used for Image Verification.
    Extracts the pair of images given in a text file
    from the specified directories.
    """

    def __init__(self, txt_file, root_dir, task='Validation', transform=None):
        self.root_dir = root_dir
        self.text_file = open(txt_file, "r")
        lines = self.text_file.read().split('\n')
        self.image_1 = list()
        self.image_2 = list()
        self.outputs = list()  # Empty for test verification
        self.task = task
        for line in lines:
            if self.task == 'Validation':
                image_1, image_2, output = line.split()
                self.image_1.append(image_1)
                self.image_2.append(image_2)
                self.outputs.append(int(output))
            else:
                image_1, image_2 = line.split()
                self.image_1.append(image_1)
                self.image_2.append(image_2)
        self.transform = transform

    def __len__(self):
        return len(self.image_1)

    def __getitem__(self, index):
        img_1 = self.root_dir + self.image_1[index]
        img_2 = self.root_dir + self.image_2[index]
        img_1 = Image.open(img_1)
        img_2 = Image.open(img_2)
        img_1 = self.transform(img_1)
        img_2 = self.transform(img_2)
        return img_1, img_2

    def getLabels(self):
        if self.task == 'Validation':
            return self.outputs
        else:
            return None


def parse_data(datadir):
    img_list = []
    for root, directories, filenames in os.walk(datadir):
        for filename in filenames:
            if filename.endswith('.jpg'):
                filei = os.path.join(root, filename)
                img_list.append(filei)
    return img_list
