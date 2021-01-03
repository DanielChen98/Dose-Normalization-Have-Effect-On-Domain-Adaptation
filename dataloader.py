from torch.utils.data import Dataset
import os
import PIL.Image as Image

class my_dataset(Dataset):
    def __init__(self, transform, path):
        self.transform = transform
        all_image = []
        all_label = []
        #label = 0
        for each_file in os.listdir(path):
            each_file_path = os.path.join(path, each_file)
            for each_image in os.listdir(each_file_path):
                #print(each_image)
                each_image_path = os.path.join(each_file_path, each_image)
                all_image.append(each_image_path)
                all_label.append(int(each_file[-3:]) - 1)
                #all_label.append(label)
            #label += 1
        self.all_label = all_label
        self.all_image = all_image

    def __getitem__(self, index):
        image_path = self.all_image[index]
        image_label = self.all_label[index]
        img = Image.open(image_path)
        if self.transform:
            img = self.transform(img)
        return img, image_label

    def __len__(self):
        return len(self.all_image)
