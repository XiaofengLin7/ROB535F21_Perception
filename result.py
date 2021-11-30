import torch
import torchvision.transforms as transforms
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from model import Yolov1
from dataset import GTADataset
from utils.utils import (
    mean_average_precision,
    get_bboxes,
    load_checkpoint,
)
from loss import YoloLoss
import time
from glob import glob
from PIL import Image

LEARNING_RATE = 2e-5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
BATCH_SIZE = 4 # 64 in original paper but I don't have that much vram, grad accum?
WEIGHT_DECAY = 0

LOAD_MODEL_FILE = './overfit20.pth.tar'

model = Yolov1(split_size=7, num_boxes=2, num_classes=3).to(DEVICE)
optimizer = optim.Adam(
    model.parameters(), lr=LEARNING_RATE, weight_decay=WEIGHT_DECAY
)
loss_fn = YoloLoss(S=7, B=2, C=3)

load_checkpoint(torch.load(LOAD_MODEL_FILE, map_location=torch.device(DEVICE)), model, optimizer)


test_files = glob('./data/test/*/*_image.jpg')
test_files.sort()
model.eval()
for f in test_files:
    image = Image.open(f)
    x = torchvision.transforms.ToTensor()(image).to(DEVICE)
    x.unsqueeze(0)
    out = model(x)
    ipdb.set_trace()
    out = out.reshape((7,7,13))




