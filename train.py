import torch
import torch.nn as nn
import torch.optim as optim
import matplotlib.pyplot as plt
from torchvision.transforms import v2
from ultralytics import YOLO



def train_one_epoch(model, train_loader, criterion, optimizer, device):
    model.train()
    train_loss = 0
    correct = 0
    total = 0

    for batch_idx, (images, labels) in enumerate(train_loader):
        print(f'Training batch {batch_idx+1}/{len(train_loader)}')
        optimizer.zero_grad()
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        train_loss += loss.item()
        
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()

    train_accuracy = 100. * correct / total
    train_loss /= len(train_loader)

    return train_loss, train_accuracy


def evaluate(model, val_loader, criterion, device):
    model.eval()
    val_loss = 0
    correct = 0
    total = 0

    with torch.no_grad():
        for batch_idx, (images, labels) in enumerate(val_loader):
            print(f'Validation batch {batch_idx+1}/{len(val_loader)}')
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            loss = criterion(outputs, labels)
            val_loss += loss.item()
            
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
        
    val_accuracy = 100. * correct / total
    val_loss /= len(val_loader)

    return val_loss, val_accuracy


def train(model, train_loader, val_loader, criterion, optimizer, num_epochs, device):
    train_losses = []
    train_accuracies = []
    val_losses = []
    val_accuracies = []

    for epoch in range(num_epochs):
        print(f'\nEpoch {epoch+1}/{num_epochs}')
        train_loss, train_accuracy = train_one_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, val_accuracy = evaluate(model, val_loader, criterion, device)
        
        train_losses.append(train_loss)
        train_accuracies.append(train_accuracy)
        val_losses.append(val_loss)
        val_accuracies.append(val_accuracy)

        print(f'Epoch {epoch+1}/{num_epochs} --> '
              f'Training Loss: {train_loss:.4f}, '
              f'Training Accuracy: {train_accuracy:.2f}%, '
              f'Validation Loss: {val_loss:.4f}, '
              f'Validation Accuracy: {val_accuracy:.2f}%')


    print(f'\nFinal Results after {num_epochs} epochs:')
    print(f'Training Loss: {train_losses[-1]:.4f}, Training Accuracy: {train_accuracies[-1]:.2f}%')
    print(f'Validation Loss: {val_losses[-1]:.4f}, Validation Accuracy: {val_accuracies[-1]:.2f}%')

    epochs = range(1, num_epochs + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, 'b-', label='Training Loss')
    plt.plot(epochs, val_losses, 'r-', label='Validation Loss')
    plt.title('Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, 'b-', label='Training Accuracy')
    plt.plot(epochs, val_accuracies, 'r-', label='Validation Accuracy')
    plt.title('Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.legend()

    plt.tight_layout()
    plt.show()





"""
img_width = 224
img_height = 224
batch_size = 10
num_classes = 300
lr = 0.0001

root_dir = 'cattle muzzle'
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
#device = torch.device('cpu')
print(device)
yolo_model = YOLO('yolov10_fine_tuned.pt').to(device)


transform = v2.Compose([
    v2.ToTensor(),
    v2.Resize((img_width, img_height))
])


train_loader, val_loader = get_loaders(root_dir = root_dir, 
                                       model = yolo_model, 
                                       batch_size = batch_size, 
                                       train_ratio=0.5,
                                       transform = transform)


clf_model = get_transfer_learning_resnet50(num_classes).to(device)
criterion = nn.CrossEntropyLoss()
optimizer = optim.Adam(clf_model.parameters(), lr=lr)
num_epochs = 1

train(clf_model, train_loader, val_loader, criterion, optimizer, num_epochs, device)



"""
