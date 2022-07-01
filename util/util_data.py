import os
import json
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset


class ODDataset(Dataset):
    def __init__(self, path_jsons, path_imgs, raw_size, post_size, transforms=None):
        self.jsons = os.listdir(path_jsons)
        self.transforms = transforms
        self.path_imgs = path_imgs
        self.path_jsons = path_jsons
        self.raw_size = raw_size
        self.post_size = post_size

    def __len__(self):
        return len(self.jsons)

    def __getitem__(self, idx):
        json_id = self.jsons[idx]
        with open(f"{self.path_jsons}/{json_id}") as f:
            file_json = json.load(f)['features']

        path_img = f"{self.path_imgs}/{file_json[0]['properties']['image_id']}"
        image = Image.open(path_img).convert('RGB')

        # (xmin, ymin, xmax, ymax) for FRCNN to work.
        # In dataset : [중심좌표 x, y, 박스크기 H, W, 회전각 θ]
        raw_boxes = np.array([obj['geometry']['coordinates'] for obj in file_json])[:, :, 0]
        boxes = np.concatenate(((raw_boxes[:, 0]-raw_boxes[:, 2]/2)[:, np.newaxis],
                                (raw_boxes[:, 1]-raw_boxes[:, 3]/2)[:, np.newaxis],
                                (raw_boxes[:, 0]+raw_boxes[:, 2]/2)[:, np.newaxis],
                                (raw_boxes[:, 1]+raw_boxes[:, 3]/2)[:, np.newaxis]), axis=1)
        # 지금 계속 음수의 bbox가 나오는 경우 있었
        boxes = np.abs(boxes)
        labels = np.array([obj['properties']['type_id'] for obj in file_json], dtype=np.int32)

        target = {'boxes': boxes, 'labels': labels}

        if self.transforms is not None:
            image = self.transforms(image)

        target['boxes'] = torch.as_tensor(target['boxes'], dtype=torch.float32)
        target['labels'] = torch.as_tensor(target['labels'], dtype=torch.int64)-1

        return image, target


def collate_fn(batch):
    return tuple(zip(*batch))