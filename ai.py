# %%
# imports
import os
import torch
import pandas as pd 
from skimage import io, transform
import numpy as np 
import matplotlib.pyplot as plt
import torchvision
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils
from torchvision.io import read_image
import PIL
from PIL import Image
from torchvision.io import read_image
from torchvision.utils import draw_bounding_boxes
from torchvision.ops import box_convert
import cv2
import albumentations as A
from albumentations.pytorch.transforms import ToTensorV2
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

from engine import train_one_epoch, evaluate
import utils
import transforms as T



# %%
root_dir = os.path.join(os.getcwd(), "data" )
test_dir = os.path.join(root_dir, "data2")
root_dir

# %%
def get_transform(train):
    if train:
        return A.Compose([
            A.HorizontalFlip(p=0.5),
            ToTensorV2(p=1.0),
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})
    else:
        return A.Compose([
            ToTensorV2(p=1.0),
        ], bbox_params={'format': 'pascal_voc', 'label_fields': ['labels']})

# %%
# Image Detection Class
class ImageDetection(Dataset):
    def __init__(self, root_dir, width, height, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        self.width = width
        self.height = height

        self.folders = os.listdir(root_dir)
        self.images = []
        self.json_data = []
        

        for folder in self.folders:
            folder_path = os.path.join(root_dir, folder)
            image_folder = os.path.join(folder_path, "images")
            json_folder = os.path.join(folder_path, "labels", "json")

            for file in os.listdir(image_folder):
                file_path = os.path.join(image_folder, file)
                self.images.append((file_path, folder))
            
            for file in os.listdir(json_folder):
                file_path = os.path.join(json_folder, file)
                self.json_data.append((file_path, folder))

    def images(self):
        return self.json_data

    def load_image(self, idx):
        img_file = self.images[idx][0]
        image = read_image(img_file)
        return Image.open(img_file).convert("RGB")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_file = self.images[idx][0]
        json_file = self.json_data[idx][0]


        #reading the images and coverting them to correct size and color
        img = cv2.imread(img_file)
        # print(img.shape)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB).astype(np.float32)
        img_res = cv2.resize(img_rgb, (self.width, self.height), cv2.INTER_AREA)
        img_res /= 255.0

        # print(img_res.shape)

        # read the json file / annotations
        df1 = pd.read_json(json_file)
       
        boxes = []
        labels = []

        wt = img.shape[1]
        ht = img.shape[0]

        pen_y  =  img_res.shape[0] / img.shape[0] 
        pen_x =  img_res.shape[1]  /img.shape[1] 

        # print("pen_x", pen_x)
        # print("pen_y", pen_y)

        for i in range(len(df1)):
            labels.append(df1["ObjectClassId"][i])
            # print( [df1["Left"][i], df1["Top"][i], df1["Right"][i], df1["Bottom"][i]])
            # print( [df1["Left"][i] * pen_y, df1["Top"][i] * pen_x, df1["Right"][i] * pen_y, df1["Bottom"][i] *pen_x])
            x1 = df1["Left"][i] *pen_x
            y1 = df1["Top"][i] * pen_y
            x2 = df1["Right"][i] * pen_x
            y2 = df1["Bottom"][i] * pen_y
            boxes.append([x1, y1, x2, y2])
        
        # print("act boxes", boxes)
        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        area = (boxes[:, 3] - boxes[:, 1]) * (boxes[:, 2] - boxes[:, 0])
        labels = torch.as_tensor(labels, dtype=torch.int64)
            
        target = {}
        target["boxes"] = boxes
        target["labels"] = labels
        target["image_id"] = torch.tensor([idx])
        target["area"] = area

        if self.transforms:
            sample = self.transforms(image = img_res, bboxes = target["boxes"], labels = labels)
            img_res = sample["image"]
            target["boxes"] = torch.Tensor(sample["bboxes"])

        return img_res, target


# data_set.__getitem__(1) 



# %%
import matplotlib.patches as patches

def plot_image_and_boxes(image, target):
    print("Image Shape is Now " , image.shape)
    fig, a = plt.subplots(1, 1)
    fig.set_size_inches(5, 5)
    a.imshow(image)
    # print("target boxes" , target["boxes"])
    for box in target["boxes"]:
        x, y , width, height  = box[0], box[1], box[2] - box[0], box[3] - box[1]
        rect  = patches.Rectangle((x, y), width, height, linewidth=2, edgecolor='r', facecolor='none')
        a.add_patch(rect)
    plt.show()



# image, target = data_set[1]
# plot_image_and_boxes(image, target)


# %%

data_set = ImageDetection(root_dir, width=280, height=240, transforms=None)



# data_set_test = ImageDetection(test_dir, width=280, height=240, transforms=None)

# torch.manual_seed(1)
# indices = torch.randperm(len(data_set)).tolist()

# test_split = 0.25
# t_size = int(len(data_set) * test_split)


train_data_set, test_data_set = torch.utils.data.random_split(data_set, [int(len(data_set)*0.75), int(len(data_set)*0.25)+1])

BATCH_SIZE = 64
NUM_WORKERS = 0
train_image_data_loader = DataLoader(dataset=train_data_set, batch_size=BATCH_SIZE, shuffle=True, num_workers=NUM_WORKERS)
test_image_data_loader = DataLoader(dataset=test_data_set, batch_size=BATCH_SIZE, shuffle=False, num_workers=NUM_WORKERS )


image, target = train_data_set[1]
plot_image_and_boxes(image, target)

len(train_image_data_loader), len(test_image_data_loader)


# %%
# pre-trained model
def get_object_detection_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn()
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model

# %%
# Training
device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')

num_classes = 2

model = get_object_detection_model(num_classes)
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9)

lr_scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

# %%
# training loop
num_epochs = 3

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, train_image_data_loader, device, epoch, print_freq=10)
    lr_scheduler.step()
    evaluate(model, test_image_data_loader, device=device)


