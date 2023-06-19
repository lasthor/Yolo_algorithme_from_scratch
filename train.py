import torch
from torch.utils.data import DataLoader
import torchvision.transforms as transforms
from torch import optim
from model import Yolov1
from loss import YoloLoss
from dataset import Data_YOLO
from utils import (
    nms,
    cellboxes_to_boxes,
    get_bboxes,
)
from tqdm import tqdm

DEVICE = "cuda" if torch.cuda.is_available else "cpu"
LEARNING_RATE = 2e-5
BATCH_SIZE = 16 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0
EPOCHS = 100
NUM_WORKERS = 2
PIN_MEMORY = True
LOAD_MODEL = False
LOAD_MODEL_FILE = "overfit.pth.tar"
IMG_DIR = "C:\\Users\\LastHour\\Desktop\\ML_folder\\Computer_vsion\YOLO\yolov5\\floss detection\\20DH-2DH-data_augmentaion\\images"
LABEL_DIR = "C:\\Users\\LastHour\\Desktop\\ML_folder\\Computer_vsion\YOLO\yolov5\\floss detection\\20DH-2DH-data_augmentaion\\labels"

transform = transforms.Compose([
    transforms.ToTensor(),
    transforms.Resize(256, interpolation=3),
    transforms.CenterCrop(224),
    
])

data_train = Data_YOLO(LABEL_DIR,IMG_DIR,transform = transform)
data_train_loader = DataLoader(data_train, batch_size = 16 , shuffle=True)


model = Yolov1(split_size=7, num_boxes=2, num_classes=2)
optimizer = optim.Adam(model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY)
loss_fn = YoloLoss()


def train_fn(train_loader, model, optimizer, loss_fn):
    loop = tqdm(train_loader, leave=True)
    mean_loss = []

    for batch_idx, (x, y) in enumerate(loop):
        x, y = x.to('cpu') , y.to('cpu')
        
        out = model(x).to('cpu')

        loss = loss_fn(out, y)
        mean_loss.append(loss.item())
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        # update progress bar
        loop.set_postfix(loss=loss.item(),out_shape=out.shape)
        break


    print(f"Mean loss was {sum(mean_loss)/len(mean_loss)},{out.shape}")
    
    

for i in range(5):
    train_fn(data_train_loader , model , optimizer , loss_fn)
    break

    
