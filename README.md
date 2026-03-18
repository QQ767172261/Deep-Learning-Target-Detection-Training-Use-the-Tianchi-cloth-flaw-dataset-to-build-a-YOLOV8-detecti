深度学习目标检测训练使用天池 布匹瑕疵数据集 来构建一个基于天池布匹瑕检测的检测系统
@[toc]
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/984fd7681d8e4ea3bbb93f19fc85d82f.png)
**天池 布匹瑕疵数据集
素色布匹5913张，34种瑕疵类型
花色布匹4371张，15种瑕疵类型
已按照瑕疵类型分好文件夹，json格式与yolo
1.天池 布匹瑕疵数据集
素色布匹5913张，34种瑕疵类型
花色布匹4371张，15种瑕疵类型
2.已按照瑕疵类型分好文件夹，标签也已分好类，json格式与yolo格式标签都有。
3.可附图片预处理代码，整理的改进点论文。实用自动脚本（自动随机划分训练集与验证集，json转yolo，EDA，生成散点图等等）。**
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/5aaf4586e36842e1ab7e94b69ac4704f.png)
1
![在这里插入图片描述](https://i-blog.csdnimg.cn/direct/999c35e32284428dad45c2cc55d39787.png)

构建一个基于天池布匹瑕疵数据集的检测系统，我们将采取以下步骤：环境设置、数据准备、模型选择与训练、模型优化、以及可视化界面的开发。

## 训练流程，

包括数据预处理、模型训练、评估以及一些优化技巧，我们将从以下几个方面进行详细说明：

### 1. 数据预处理

首先，我们需要定义如何加载和预处理数据。这包括图像的读取、转换为张量格式、应用必要的变换（如归一化、数据增强等）。

```python
import os
import cv2
import json
import torch
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T

class FabricDefectDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transforms=None):
        self.root_dir = root_dir
        self.transforms = transforms
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

    def __getitem__(self, idx):
        img_info = self.annotations[idx]
        img_path = os.path.join(self.root_dir, img_info['filename'])
        image = cv2.imread(img_path)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

        boxes = []
        labels = []
        for obj in img_info['objects']:
            bbox = obj['bounding_box']
            boxes.append([bbox[0], bbox[1], bbox[2] + bbox[0], bbox[3] + bbox[1]]) # x_min, y_min, x_max, y_max
            labels.append(obj['label_id'] + 1)  # 加1是因为背景类别为0

        boxes = torch.as_tensor(boxes, dtype=torch.float32)
        labels = torch.as_tensor(labels, dtype=torch.int64)

        target = {}
        target["boxes"] = boxes
        target["labels"] = labels

        if self.transforms is not None:
            image, target = self.transforms(image, target)

        return T.ToTensor()(image), target

    def __len__(self):
        return len(self.annotations)

def get_transform(train):
    transforms = []
    transforms.append(T.ToTensor())
    if train:
        transforms.append(T.RandomHorizontalFlip(0.5))  # 数据增强：随机水平翻转
    return T.Compose(transforms)
```

### 2. 模型配置与加载

接下来是选择合适的模型并加载预训练权重，这里我们使用Faster R-CNN。

```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
```

### 3. 训练流程

定义完整的训练流程，包括训练循环、学习率调整、保存最佳模型等。

```python
from references.detection.engine import train_one_epoch, evaluate
from references.detection.utils import collate_fn
import torch.optim as optim

# 假设已经创建了dataset_train和dataset_val
data_loader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)
data_loader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes=35)  # 包括背景在内的35个类别
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

lr_scheduler = optim.lr_scheduler.StepLR(optimizer, step_size=3, gamma=0.1)

num_epochs = 10
best_model_wts = model.state_dict()
best_acc = 0.0

for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, data_loader_train, device, epoch, print_freq=10)
    lr_scheduler.step()

    # Evaluate on validation set
    coco_evaluator = evaluate(model, data_loader_val, device=device)

    # Deep copy the model
    if coco_evaluator.coco_eval['bbox'].stats[0] > best_acc:
        best_acc = coco_evaluator.coco_eval['bbox'].stats[0]
        best_model_wts = model.state_dict()

    torch.save(model.state_dict(), f'model_epoch_{epoch}.pth')

model.load_state_dict(best_model_wts)
torch.save(model.state_dict(), 'best_model.pth')
```

### 4. 可视化界面

最后，我们可以使用Streamlit来构建一个简单的Web应用，用于展示预测结果。

```python
import streamlit as st
from PIL import Image
import torchvision.transforms as T
import torch

st.title("布匹瑕疵检测系统")

@st.cache(allow_output_mutation=True)
def load_model():
    model = get_model(num_classes=35)
    model.load_state_dict(torch.load('best_model.pth', map_location=torch.device('cpu')))
    model.eval()
    return model

model = load_model()

transform = T.Compose([T.ToTensor()])

uploaded_file = st.file_uploader("上传一张图片...", type=["jpg", "jpeg"])
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(tensor)

    st.image(image, caption='上传的图片.', use_column_width=True)
    st.write("预测结果:")
    for box, label in zip(prediction[0]['boxes'].numpy(), prediction[0]['labels'].numpy()):
        st.write(f"瑕疵位置: {box}, 类别ID: {label}")
```

从数据加载、模型训练到可视化的一个完整流程。注意替换路径和参数为实际值，并根据实际情况调整代码细节，比如数据集路径、模型保存路径等。此外，`references/detection/`下的文件需要从官方PyTorch检测示例库中获取，确保它们正确地集成到您的项目中。
### 1. 环境设置

首先确保安装了必要的库：

```bash
pip install torch torchvision opencv-python numpy matplotlib pycocotools jupyterlab
```

### 2. 数据准备

假设数据已按照瑕疵类型分好文件夹，且有JSON和YOLO两种格式的标注。我们需要创建一个自定义的Dataset类来加载这些数据。

```python
import os
import cv2
import torch
from torch.utils.data import Dataset, DataLoader
import json

class FabricDefectDataset(Dataset):
    def __init__(self, root_dir, annotation_file, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        with open(annotation_file, 'r') as f:
            self.annotations = json.load(f)

    def __len__(self):
        return len(self.annotations)

    def __getitem__(self, idx):
        img_name = os.path.join(self.root_dir, self.annotations[idx]['filename'])
        image = cv2.imread(img_name)
        image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
        boxes = []
        labels = []
        for obj in self.annotations[idx]['objects']:
            bbox = obj['bounding_box']
            boxes.append([bbox[0], bbox[1], bbox[2], bbox[3]])
            labels.append(obj['label_id'])

        target = {
            'boxes': torch.tensor(boxes, dtype=torch.float32),
            'labels': torch.tensor(labels, dtype=torch.int64)
        }

        if self.transform:
            image, target = self.transform(image, target)

        return image, target
```

### 3. 模型选择与训练

考虑到任务是目标检测，Faster R-CNN是一个合适的选择。

```python
import torchvision
from torchvision.models.detection.faster_rcnn import FastRCNNPredictor

def get_model(num_classes):
    model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
    in_features = model.roi_heads.box_predictor.cls_score.in_features
    model.roi_heads.box_predictor = FastRCNNPredictor(in_features, num_classes)
    return model
```

### 4. 训练流程

使用`torchvision`提供的工具进行训练。

```python
from references.detection.engine import train_one_epoch, evaluate
from references.detection.utils import collate_fn

dataset_train = FabricDefectDataset('path/to/train', 'annotations.json')
dataloader_train = DataLoader(dataset_train, batch_size=2, shuffle=True, num_workers=4, collate_fn=collate_fn)

dataset_val = FabricDefectDataset('path/to/val', 'annotations.json')
dataloader_val = DataLoader(dataset_val, batch_size=1, shuffle=False, num_workers=4, collate_fn=collate_fn)

device = torch.device('cuda') if torch.cuda.is_available() else torch.device('cpu')
model = get_model(num_classes=35)  # 包括背景在内的35个类别
model.to(device)

params = [p for p in model.parameters() if p.requires_grad]
optimizer = torch.optim.SGD(params, lr=0.005, momentum=0.9, weight_decay=0.0005)

num_epochs = 10
for epoch in range(num_epochs):
    train_one_epoch(model, optimizer, dataloader_train, device, epoch, print_freq=10)
    evaluate(model, dataloader_val, device=device)
```

### 5. 模型优化

- **学习率调整**：根据训练情况动态调整学习率。
- **数据增强**：增加更多的数据增强技术，如随机裁剪、旋转等。
- **提前停止**：监控验证集上的性能，如果在一定轮次内没有提升，则提前结束训练。

### 6. 可视化界面

可以使用Streamlit或Flask快速搭建一个简单的Web应用用于展示结果。

以Streamlit为例：

```python
import streamlit as st
import torch
from PIL import Image
import torchvision.transforms as T

st.title("布匹瑕疵检测系统")

# 加载预训练模型
model = get_model(num_classes=35)
model.load_state_dict(torch.load('model.pth'))
model.eval()

transform = T.Compose([T.ToTensor()])

uploaded_file = st.file_uploader("Choose an image...", type="jpg")
if uploaded_file is not None:
    image = Image.open(uploaded_file).convert("RGB")
    tensor = transform(image).unsqueeze(0)

    with torch.no_grad():
        prediction = model(tensor)

    st.image(image, caption='Uploaded Image.', use_column_width=True)
    st.write("预测结果:")
    for box, label in zip(prediction[0]['boxes'], prediction[0]['labels']):
        st.write(f"瑕疵位置: {box.tolist()}, 类别ID: {label.item()}")
```

以上提供了从数据准备到模型部署的基本框架，可以根据具体需求进一步定制和优化。注意替换路径和参数为实际值。
