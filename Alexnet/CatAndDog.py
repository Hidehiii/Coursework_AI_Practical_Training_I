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

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# 定义数据集路径
data_dir = './Data/microsoft-catsvsdogs-dataset/versions/1/PetImages/'  # 数据集路径
cat_dir = os.path.join(data_dir, 'Cat')  # 猫的图像目录
dog_dir = os.path.join(data_dir, 'Dog')  # 狗的图像目录

def valid_image_file(file_path):
    """检查文件是否为有效的图像文件且大小不为零"""
    try:
        with Image.open(file_path) as img:
            if img.size[0] == 0 or img.size[1] == 0:
                return False
            img.verify()  # 验证图像
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

# 确认有效的图像数量
print(f"有效猫图像数量: {len(cat_images)}")
print(f"有效狗图像数量: {len(dog_images)}")

# 2. 切分训练集和测试集
cat_train, cat_test = train_test_split(cat_images, test_size=0.2, random_state=42)
dog_train, dog_test = train_test_split(dog_images, test_size=0.2, random_state=42)


# 3. 创建自定义数据集
class CatsAndDogsDataset(Dataset):
    def __init__(self, cat_images, dog_images, transform=None):
        self.images = cat_images + dog_images
        self.labels = [0] * len(cat_images) + [1] * len(dog_images)  # 0: cat, 1: dog
        self.transform = transform

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        img_path = self.images[idx]
        image = Image.open(img_path).convert('RGB')  # 确保读入RGB
        label = self.labels[idx]

        if self.transform:
            image = self.transform(image)

        return image, label

    # 4. 数据预处理和加载


transform = transforms.Compose([
    transforms.Resize((224, 224)),  # 调整图像为224x224
    transforms.ToTensor(),  # 转换为Tensor
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])  # 归一化
])

# 创建训练集和测试集实例
train_dataset = CatsAndDogsDataset(cat_train, dog_train, transform=transform)
test_dataset = CatsAndDogsDataset(cat_test, dog_test, transform=transform)

train_loader = DataLoader(train_dataset, batch_size=32, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False)


# 5. 可视化数据
def imshow(img):
    img = img / 2 + 0.5  # 去归一化
    npimg = img.numpy()
    plt.imshow(np.transpose(npimg, (1, 2, 0)))
    plt.axis('off')


# 获取一些随机图像
dataiter = iter(train_loader)
images, labels = next(dataiter)

# 展示图像
imshow(torchvision.utils.make_grid(images[:6]))  # 展示前6张图片
plt.title('Sample Images from Cats and Dogs Dataset')
plt.show()


# 6. 定义AlexNet模型
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
            nn.Linear(128 * 56 * 56, 256),  # 根据输入的图像尺寸调整
            nn.ReLU(inplace=True),
            nn.Linear(256, 2),  # 输出2个类别
        )

    def forward(self, x):
        x = self.features(x)
        x = self.classifier(x)
        return x

    # 7. 训练函数


def train_model(model, dataloader, criterion, optimizer, num_epochs=5):
    model.train()
    model.to(device)

    train_losses = []
    train_accuracies = []


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

        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {running_loss / len(dataloader):.4f}")

    return train_losses, train_accuracies

    # 8. 训练模型


model = CustomAlexNet()
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.Adam(model.parameters(), lr=0.001)
train_losses, train_accuracies = train_model(model, train_loader, criterion, optimizer, num_epochs=5)


# 9. 可视化训练过程
def plot_training_history(train_losses, train_accuracies):
    epochs = range(1, len(train_losses) + 1)

    plt.figure(figsize=(12, 5))

    # 绘制训练损失
    plt.subplot(1, 2, 1)
    plt.plot(epochs, train_losses, label='Training Loss')
    plt.title('Training Loss')
    plt.xlabel('Epochs')
    plt.ylabel('Loss')
    plt.grid(True)

    # 绘制训练准确率
    plt.subplot(1, 2, 2)
    plt.plot(epochs, train_accuracies, label='Training Accuracy', color='orange')
    plt.title('Training Accuracy')
    plt.xlabel('Epochs')
    plt.ylabel('Accuracy (%)')
    plt.grid(True)

    plt.tight_layout()
    plt.show()


plot_training_history(train_losses, train_accuracies)

# 10. 测试模型并比较训练效果（可选）
# 在测试集上评估模型准确率
def evaluate_model(model, test_loader):
    model.eval()
    model.to(device)
    correct = 0
    total = 0
    accuracies = []
    with torch.no_grad():
        for images, labels in test_loader:
            images, labels = images.to(device), labels.to(device)
            outputs = model(images)
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            accuracies.append(100 * correct / total)
    print(f'Accuracy: {100 * correct / total:.2f}%')
    return accuracies


test_accuracy = evaluate_model(model, test_loader)

# 可视化测试结果
def plot_test_results(train_accuracies, test_accuracy):
    epochs = range(1, min(len(train_accuracies), len(test_accuracy)) + 1)

    plt.figure(figsize=(6, 6))
    plt.plot(epochs, train_accuracies[:len(epochs)], label='Training Accuracy', color='orange')
    plt.plot(epochs, test_accuracy[:len(epochs)], label='Test Accuracy', color='blue')
    # 显示label
    plt.legend()
    plt.title('Model Accuracy Comparison')
    plt.ylabel('Accuracy (%)')
    plt.show()

plot_test_results(train_accuracies, test_accuracy)

# 11. 测试模型并比较结果
def visualize_test_results(model, test_loader, num_images=6):
    model.eval()
    model.to(device)
    images, labels = [], []

    for _ in range(num_images):
        idx = random.randint(0, len(test_loader.dataset) - 1)
        img, label = test_loader.dataset[idx]  # 获取图像、标签路径
        images.append(img.unsqueeze(0))  # 增加批处理维度
        labels.append(label)

    images = torch.cat(images)  # 合并图像
    labels = torch.tensor(labels)  # 转换为tensor

    with torch.no_grad():
        images, labels = images.to(device), labels.to(device)
        outputs = model(images)
        _, predicted = torch.max(outputs.data, 1)

    # 可视化结果
    plt.figure(figsize=(15, 5))
    for i in range(num_images):
        plt.subplot(2, num_images, i + 1)
        img = images[i].cpu().numpy().transpose((1, 2, 0))  # 转换为HWC格式
        img = img * 0.5 + 0.5  # 反归一化
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'Pred: {"Dog" if predicted[i].item() else "Cat"}')

        # 显示正确标签
        plt.subplot(2, num_images, num_images + i + 1)
        plt.imshow(img)
        plt.axis('off')
        plt.title(f'True: {"Dog" if labels[i].item() else "Cat"}')

    plt.tight_layout()
    plt.show()

visualize_test_results(model, test_loader)