import torch
import torch.nn as nn
from torchvision.transforms import v2
from ultralytics import YOLO
from torchvision.models import resnet50, ResNet50_Weights


def get_transfer_learning_resnet50(num_classes):
    resnet50_model = resnet50(weights=ResNet50_Weights.DEFAULT)

    for param in resnet50_model.parameters():
        param.requires_grad = False

    resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, num_classes)

    return resnet50_model

    """
    model_with_softmax = nn.Sequential(
        resnet50_model,
        nn.Softmax(dim=1) 
    )
    
    return model_with_softmax
    """



def get_finetuned_resnet50_last_conv(num_classes):
    resnet50_model = resnet50(weights=ResNet50_Weights.DEFAULT)

    for param in resnet50_model.parameters():
        param.requires_grad = False

    last_conv_layer = resnet50_model.layer4[2].conv3 
    for param in last_conv_layer.parameters():
        param.requires_grad = True

    resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, num_classes)

    return resnet50_model

"""
    model_with_softmax = nn.Sequential(
        resnet50_model,
        nn.Softmax(dim=1)  
    )
    
    return model_with_softmax
   """



def get_finetuned_resnet50_last_three_convs(num_classes):
    resnet50_model = resnet50(weights=ResNet50_Weights.DEFAULT)
    
    for param in resnet50_model.parameters():
        param.requires_grad = False

    last_block = resnet50_model.layer4[2] 
    for i in range(3): 
        conv_layer = last_block.__getattr__(f'conv{i+1}') 
        for param in conv_layer.parameters():
            param.requires_grad = True

    resnet50_model.fc = nn.Linear(resnet50_model.fc.in_features, num_classes)
    
    return resnet50_model

    """
    model_with_softmax = nn.Sequential(
        resnet50_model,
        nn.Softmax(dim=1) 
    )
    
    return model_with_softmax
    """



class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, in_channels, out_channels, stride=1):
        super(BasicBlock, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, in_channels, out_channels, stride=1):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.conv2 = nn.Conv2d(out_channels, out_channels, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.conv3 = nn.Conv2d(out_channels, out_channels * self.expansion, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(out_channels * self.expansion)
        self.relu = nn.ReLU(inplace=True)

        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != self.expansion * out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv2d(in_channels, self.expansion * out_channels, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(self.expansion * out_channels)
            )

    def forward(self, x):
        out = self.relu(self.bn1(self.conv1(x)))
        out = self.relu(self.bn2(self.conv2(out)))
        out = self.bn3(self.conv3(out))
        out += self.shortcut(x)
        out = self.relu(out)
        return out
    

class ResNet(nn.Module):
    def __init__(self, block, layers, num_classes=1000):
        super(ResNet, self).__init__()
        self.in_channels = 64

        self.conv1 = nn.Conv2d(3, 64, kernel_size=7, stride=2, padding=3, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)

        self.layer1 = self._make_layer(block, 64, layers[0])
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2)
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2)
        self.layer4 = self._make_layer(block, 512, layers[3], stride=2)

        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.fc = nn.Linear(512 * block.expansion, num_classes)

    def _make_layer(self, block, out_channels, blocks, stride=1):
        layers = []
        layers.append(block(self.in_channels, out_channels, stride))
        self.in_channels = out_channels * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_channels, out_channels))
        return nn.Sequential(*layers)

    def forward(self, x):
        x = self.relu(self.bn1(self.conv1(x)))
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        x = self.avgpool(x)
        x = torch.flatten(x, 1)
        x = self.fc(x)

        return x
    

def get_resnet50(num_classes=1000):
    return ResNet(Bottleneck, [3, 4, 6, 3], num_classes)




def count_parameters(model):
    return sum(p.numel() for p in model.parameters())


def count_trainable_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def print_model_info(model, model_name):
    total_params = count_parameters(model)
    trainable_params = count_trainable_parameters(model)
    
    print(f"\n{model_name}:")
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    print(f"Percentage of trainable parameters: {trainable_params/total_params*100:.2f}%")




"""
num_classes = 300
models = {
    "Scratch" : get_resnet50(num_classes),
    "Transfer Learning": get_transfer_learning_resnet50(num_classes),
    "Fine-tuned (Last Conv)": get_finetuned_resnet50_last_conv(num_classes),
    "Fine-tuned (Last 3 Convs)": get_finetuned_resnet50_last_three_convs(num_classes)
}

for name, model in models.items():
    print_model_info(model, name)
    
"""


"""
from PIL import Image

def inference(img_path, transform, yolo_model, clf_model, device):
    img = Image.open(img_path)
    img_tensor = transform(img)

    results = yolo_model(img_tensor)  

    if len(results) > 0:
        bbox = results[0].boxes.xyxy[0] 

        xmin, ymin, xmax, ymax = map(int, bbox)
        cropped_img = img.crop((xmin, ymin, xmax, ymax))

        cropped_img_tensor = transform(cropped_img)
        cropped_img_tensor = cropped_img_tensor.to(device)
        cropped_img_tensor = cropped_img_tensor.unsqueeze(0) 

        output = clf_model(cropped_img_tensor)
        print(output)
        #return output
        return torch.argmax(output).item()


img_width = 224
img_height = 224
img_path = r'cattle muzzle\069\IMG_0906.JPG'
transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((img_width, img_height))
])
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
yolo_model = YOLO('yolov10_fine_tuned.pt')
clf_model = torch.load('resnet_transfer_learning.pth')
print(inference(img_path, transform, yolo_model, clf_model, device))
"""