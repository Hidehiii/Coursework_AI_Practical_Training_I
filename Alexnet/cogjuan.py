import os
import random
import numpy as np
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
import torchvision.transforms as transforms
import torchvision.datasets as datasets
import torch
import torchvision
import torch.nn as nn
import torchvision.models as models
from torch.utils.data import DataLoader, Dataset
from PIL import Image
from matplotlib import rcParams
rcParams['font.sans-serif'] = ['SimHei']  # 使用黑体
rcParams['axes.unicode_minus'] = False  # 正确显示负号

# 检查设备是否为 GPU
assert torch.cuda.is_available(), "当前没有检测到可用的 GPU，请检查 CUDA 是否正确安装。"
device = torch.device('cuda')

# 定义数据集路径
data_dir = './Data/microsoft-catsvsdogs-dataset/versions/1/PetImages/'
cat_dir = os.path.join(data_dir, 'Cat')
dog_dir = os.path.join(data_dir, 'Dog')

def valid_image_file(file_path):
    """检查文件是否为有效的图像且大小非零。"""
    try:
        with Image.open(file_path) as img:
            if img.size[0] == 0 or img.size[1] == 0:
                return False
            img.verify()
        return True
    except Exception:
        return False

cat_images = [
    os.path.join(cat_dir, img) for img in os.listdir(cat_dir)
    if img.lower().endswith(('.jpg', '.png')) and valid_image_file(os.path.join(cat_dir, img))
]
dog_images = [
    os.path.join(dog_dir, img) for img in os.listdir(dog_dir)
    if img.lower().endswith(('.jpg', '.png')) and valid_image_file(os.path.join(dog_dir, img))
]

print(f"有效猫图像数量: {len(cat_images)}")
print(f"有效狗图像数量: {len(dog_images)}")

cat_train, cat_test = train_test_split(cat_images, test_size=0.2, random_state=42)
dog_train, dog_test = train_test_split(dog_images, test_size=0.2, random_state=42)

class CatsAndDogsDataset(Dataset):
    def __init__(self, cat_images, dog_images, transform=None):
        self.images = cat_images + dog_images
        self.labels = [0] * len(cat_images) + [1] * len(dog_images)
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

transform = transforms.Compose([
    transforms.Resize((224, 224)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

train_dataset = CatsAndDogsDataset(cat_train, dog_train, transform=transform)
test_dataset = CatsAndDogsDataset(cat_test, dog_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=256, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=256, shuffle=False)

def imshow(img):
    img = img / 2 + 0.5
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')

# 检查数据是否正确加载
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 将数据移动到 GPU
images, labels = images.to(device), labels.to(device)

# 检查是否成功移动
assert images.is_cuda, "图像数据未加载到 GPU 上！"
assert labels.is_cuda, "标签数据未加载到 GPU 上！"

imshow(torchvision.utils.make_grid(images[:6].cpu()))  # 必须转回 CPU 才能显示
plt.title('猫狗数据集的样本图片')
plt.show()

class CustomAlexNet(nn.Module):
    def __init__(self):
        super(CustomAlexNet, self).__init__()
        self.features = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Conv2d(64, 128, kernel_size=(3, 3), stride=1, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=2, stride=2),
            nn.Flatten(),
        )
        self.classifier = nn.Sequential(
            nn.Linear(128 * 56 * 56, 256),
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

def train_model(model, dataloader, criterion, optimizer, num_epochs=10, save_dir='models'):
    model.train()
    model.to(device)

    train_losses = []
    train_accuracies = []

    if not os.path.exists(save_dir):
        os.makedirs(save_dir)

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for images, labels in dataloader:
            images, labels = images.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(images)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        epoch_loss = running_loss / len(dataloader)
        epoch_accuracy = 100 * correct / total

        train_losses.append(epoch_loss)
        train_accuracies.append(epoch_accuracy)

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {epoch_loss:.4f}, Accuracy: {epoch_accuracy:.2f}%")

        # 保存最新的模型，保留最新的五个
        model_path = os.path.join(save_dir, f'model_epoch_{epoch + 1}.pth')
        torch.save(model.state_dict(), model_path)

        # 清理旧模型
        saved_models = sorted([f for f in os.listdir(save_dir) if f.startswith('model_epoch_')])
        while len(saved_models) > 5:
            os.remove(os.path.join(save_dir, saved_models.pop(0)))

    return train_losses, train_accuracies

model = CustomAlexNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.0003)

train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs=20, save_dir='models')

def plot_training_history(train_losses, train_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='训练损失')
    plt.title('训练损失')
    plt.xlabel('轮次')
    plt.ylabel('损失')
    plt.grid(True)

    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='训练准确率', color='orange')
    plt.title('训练准确率')
    plt.xlabel('轮次')
    plt.ylabel('准确率 (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()

plot_training_history(train_losses, train_accuracies)

def evaluate_model(model, test_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
    accuracy = 100 * correct / total
    print(f'准确率: {accuracy:.2f}%')
    return accuracy

evaluate_model(model, test_loader)

# 可选: 加载保存的模型并评估
model.load_state_dict(torch.load('models/model_epoch_5.pth'))
evaluate_model(model, test_loader)