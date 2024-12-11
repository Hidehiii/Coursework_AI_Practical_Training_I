import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

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

# PCA降维（任选您的要求）
pca = PCA(n_components=10)  # 选择10个主成分
X_pca = pca.fit_transform(X_scaled)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X_pca, y, test_size=0.2, random_state=42)

# 转换为张量类型
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train).long()  # 确保目标为长整型
X_test_tensor = torch.from_numpy(X_test).float()
y_test_tensor = torch.from_numpy(y_test).long()  # 确保目标为长整型


# 定义全连接神经网络模型
class MLPModel(nn.Module):
    def __init__(self, num_classes=2):
        super(MLPModel, self).__init__()
        self.fc1 = nn.Linear(10, 64)  # 输入特征数为10，输出为64个神经元
        self.fc2 = nn.Linear(64, 32)  # 隐藏层
        self.fc3 = nn.Linear(32, num_classes)  # 输出层

    def forward(self, x):
        x = torch.relu(self.fc1(x))  # ReLU激活
        x = torch.relu(self.fc2(x))  # ReLU激活
        return self.fc3(x)

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
    model = MLPModel()
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