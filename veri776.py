from torch.utils.data import Dataset, DataLoader
import xml.etree.ElementTree as ET
from PIL import Image
import numpy as np
from collections import defaultdict
import torch


import os


def parse_xml(xml_path):
    with open(xml_path) as f:
        et = ET.fromstring(f.read())

        image_paths = []
        vehicle_ids = []
        class_map = dict()
        cur_class = 0


        for item in et.iter('Item'):
            image_paths.append(item.attrib['imageName'])

            vehicle_id = int(item.attrib['vehicleID'])
            vehicle_ids.append(vehicle_id)

            if vehicle_id not in class_map:
                class_map[vehicle_id] = cur_class
                cur_class += 1

        return image_paths, vehicle_ids, class_map


def get_veri776(veri776_path, num_workers=6, batch_size=32, training_transform=None, test_transform=None):
    
    train_img_paths, train_vehicle_ids, train_class_map = parse_xml(os.path.join(veri776_path, 'train_label.xml'))
    train_img_paths = [os.path.join(veri776_path, 'image_train', path) for path in train_img_paths]
    train_set = Veri776(train_img_paths, train_vehicle_ids, train_class_map, training_transform)


    test_img_paths, test_vehicle_ids, test_class_map = parse_xml(os.path.join(veri776_path, 'test_label.xml'))
    test_img_paths = [os.path.join(veri776_path, 'image_test', path) for path in train_img_paths]
    test_set = Veri776(test_img_paths, test_vehicle_ids, test_class_map, test_transform)

    return DataLoader(train_set, num_workers=num_workers, batch_size=batch_size), DataLoader(test_set, num_workers=num_workers, batch_size=batch_size)
   

class Veri776(Dataset):
    def __init__(self, img_paths, vehicle_ids, class_map, transform):
        self.img_paths = np.array(img_paths)
        self.vehicle_ids = np.array(vehicle_ids)
        self.class_map = class_map # map the vehicle id to the label used for classification
        self.transform = transform

        self.class_tree = self.build_class_tree(vehicle_ids, class_map, img_paths) # maps the class to a list of images which has that class

    def build_class_tree(self, vehicle_ids, class_map, img_paths):
        class_tree = defaultdict(list)
        for id, path in zip(vehicle_ids, img_paths):
            class_tree[class_map[id]].append(path) 

        return class_tree
    

        

    def __len__(self):
        return len(self.img_paths)
    

    def __getitem__(self, index):
        anchor_img = Image.open(self.img_paths[index])

        label = self.class_map[self.vehicle_ids[index]]
        positive_img_path = np.random.choice(self.class_tree[label])
        positive_img = Image.open(positive_img_path)



        negative_img_class = self.random_number_except(0, len(self.class_map), label)
        negative_img_path = np.random.choice(self.class_tree[negative_img_class])
        negative_img = Image.open(negative_img_path)

        if self.transform is not None:
            positive_img = self.transform(positive_img)
            negative_img = self.transform(negative_img)
            anchor_img = self.transform(anchor_img)


        return torch.stack((anchor_img, positive_img, negative_img), dim=0), torch.tensor([label, label, negative_img_class])
    


    def random_number_except(self, range_start, range_end, excluded_number):
        numbers = list(range(range_start, range_end))  # Create a list of numbers in the specified range
        numbers.remove(excluded_number)  # Remove the excluded number from the list
        return np.random.choice(numbers)
    

if __name__ == '__main__':
    import Transforms
    import torchvision
    import einops
    veri776_train, _ = get_veri776('../veri776', training_transform=Transforms.get_training_transform(), batch_size=10, num_workers=1)
    X, y = next(iter(veri776_train))
    X = einops.rearrange(X, 'b t c h w -> (b t) c h w')
    print(len(X))
    grid_image = torchvision.utils.make_grid(X, nrow=3)
    torchvision.utils.save_image(grid_image, 'test.png')