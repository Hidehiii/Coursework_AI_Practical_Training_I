import torch
import torch.nn as nn
import torch.optim as optim
from torchvision import models
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import torch.nn.functional as F

device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


# 加载乳腺癌数据集
data = load_breast_cancer()
X, y = data.data, data.target


# 数据分析可视化
def visualize_features(X, y):
    fig, axes = plt.subplots(5, 6, figsize=(15, 10))
    ax = axes.ravel()
    for i in range(30):
        ax[i].hist(X[y == 0, i], bins=30, color='r', alpha=0.5, label='恶性')
        ax[i].hist(X[y == 1, i], bins=30, color='b', alpha=0.5, label='良性')
        ax[i].set_title(data.feature_names[i])
        ax[i].legend()
    plt.tight_layout()
    plt.show()


visualize_features(X, y)

# 数据预处理
scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)

# PCA降维（可选）
pca = PCA(n_components=10)  # 选择10个主成分
X_pca = pca.fit_transform(X_scaled)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 转换为张量类型并调整形状为4D
X_train_tensor = torch.from_numpy(X_train).float().view(-1, 1, 1, 10)  # 4D: batch size, channels, height, width
y_train_tensor = torch.from_numpy(y_train).long()
X_test_tensor = torch.from_numpy(X_test).float().view(-1, 1, 1, 10)  # 4D: batch size, channels, height, width
y_test_tensor = torch.from_numpy(y_test).long()

# 用cuda GPU加速
X_train_tensor = X_train_tensor.to(device)
y_train_tensor = y_train_tensor.to(device)
X_test_tensor = X_test_tensor.to(device)
y_test_tensor = y_test_tensor.to(device)


# 定义修改后的AlexNet模型
class ModifiedAlexNet(nn.Module):
    def __init__(self):
        super(ModifiedAlexNet, self).__init__()
        self.alexnet = models.alexnet(weights='DEFAULT')

        # 修改输入层 - 第一个卷积层
        self.alexnet.features[0] = nn.Conv2d(1, 64, kernel_size=(1, 1), stride=(1, 1), padding=(0, 0))
        # 之后的卷积层调整使其兼容小输入
        self.alexnet.features[1] = nn.BatchNorm2d(64)  # 保留BN层
        self.alexnet.features[2] = nn.ReLU(inplace=True)
        # self.alexnet.features[3] = nn.MaxPool2d(kernel_size=1)  # 更改池化层
        # 替换最后的分类器
        self.alexnet.classifier[6] = nn.Linear(4096, 2)  # 输出2个类别

    def forward(self, x):
        return self.alexnet(x)

    # 训练模型函数


def train_model(model, X_train_tensor, y_train_tensor, num_epochs=10, batch_size=64, learning_rate=0.001):
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=learning_rate, momentum=0.9)

    train_loss_history = []
    accuracy_history = []

    for epoch in range(num_epochs):
        running_loss = 0.0
        correct = 0
        total = 0
        for i in range(0, len(X_train_tensor), batch_size):
            inputs = X_train_tensor[i:i + batch_size]
            labels = y_train_tensor[i:i + batch_size]

            inputs = inputs.repeat(1, 1, 10, 1)  # 重复3次，使其符合AlexNet输入要求
            #inputs = F.interpolate(inputs, size=(11, 11), mode='bilinear', align_corners=False)

            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()

        train_loss_history.append(running_loss / (len(X_train_tensor) / batch_size))
        accuracy_history.append(correct / total)

    return train_loss_history, accuracy_history


# 定义多个模型并比较结果
learning_rates = [0.001, 0.01]
num_epochs = 10
results = {}

for lr in learning_rates:
    model = ModifiedAlexNet()
    model.to(device)
    train_loss, train_accuracy = train_model(model, X_train_tensor, y_train_tensor, num_epochs=num_epochs,
                                             learning_rate=lr)
    results[lr] = {'loss': train_loss, 'accuracy': train_accuracy}

# 可视化训练损失与准确率
fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 5))

for lr, result in results.items():
    ax1.plot(range(1, num_epochs + 1), result['loss'], label=f'LR: {lr}')
    ax2.plot(range(1, num_epochs + 1), result['accuracy'], label=f'LR: {lr}')

ax1.set_xlabel('Epochs')
ax1.set_ylabel('Loss')
ax1.set_title('Training Loss over epochs')
ax1.legend()

ax2.set_xlabel('Epochs')
ax2.set_ylabel('Accuracy')
ax2.set_title('Training Accuracy over epochs')
ax2.legend()

plt.tight_layout()
plt.show()