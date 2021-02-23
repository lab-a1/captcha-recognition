import torch
import string
import os
from torch.utils.data import Dataset
import cv2


class CaptchasDataset(Dataset):
    def __init__(
        self, images_path, ground_truth, transpose=True, label_as_str=False
    ):
        self.ground_truth = ground_truth
        self.images_path = images_path
        self.transpose = transpose
        self.label_as_str = label_as_str
        self.label_to_index = {
            char: i for i, char in enumerate(string.printable)
        }
        self.index_to_label = {
            i: char for i, char in enumerate(string.printable)
        }

    def __len__(self):
        return len(self.ground_truth)

    def __getitem__(self, index):
        if isinstance(index, slice):
            return [self[i] for i in range(index.start, index.stop)]
        elif torch.is_tensor(index):
            index = index.tolist()

        item = self.ground_truth.loc[index]
        image_path = os.path.join(self.images_path, item[0])
        image = cv2.imread(image_path)
        # OpenCV reads by default as BGR.
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        if self.transpose:
            # 200x50x3 => 3x200x50
            image = image.transpose((2, 0, 1))
        if self.label_as_str:
            return image, item[1]
        return image, [self.label_to_index[x] for x in item[1]]
