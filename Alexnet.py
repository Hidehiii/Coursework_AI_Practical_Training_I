import torch
import torch.nn as nn
import torchvision.models as models
from sklearn.datasets import load_breast_cancer
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import matplotlib.pyplot as plt

# 加载乳腺癌数据集
data = load_breast_cancer()
X, y = data.data, data.target

# 数据分析可视化
fig, axes = plt.subplots(5, 6, figsize=(15, 10))
ax = axes.ravel()
for i in range(30):
    ax[i].hist(X[y == 0, i], bins=30, color='r', alpha=0.5, label='恶性')
    ax[i].hist(X[y == 1, i], bins=30, color='b', alpha=0.5, label='良性')
    ax[i].set_title(data.feature_names[i])
plt.tight_layout()
plt.show()

# 数据预处理
scaler = StandardScaler()
X = scaler.fit_transform(X)

# 划分训练集和测试集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 转换为张量类型
X_train_tensor = torch.from_numpy(X_train).float()
y_train_tensor = torch.from_numpy(y_train)

# 构建AlexNet模型
class AlexNetModel(nn.Module):
    def __init__(self, num_classes=2):
        super(AlexNetModel, self).__init__()
        self.alexnet = models.alexnet(pretrained=True)
        self.alexnet.classifier[6] = nn.Linear(4096, num_classes)

    def forward(self, x):
        return self.alexnet(x)

model = AlexNetModel()

# 定义损失函数和优化器
criterion = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model.parameters(), lr=0.001, momentum=0.9)

# 训练模型
train_loss_history = []
num_epochs = 10
batch_size = 64

for epoch in range(num_epochs):
    running_loss = 0.0
    for i in range(0, len(X_train_tensor), batch_size):
        inputs = X_train_tensor[i:i+batch_size]
        labels = y_train_tensor[i:i+batch_size]

        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()

        running_loss += loss.item()
    train_loss_history.append(running_loss / (len(X_train_tensor) / batch_size))

# 可视化训练损失
plt.plot(range(1, num_epochs + 1), train_loss_history)
plt.xlabel('Epochs')
plt.ylabel('Loss')
plt.title('Training Loss over epochs')
plt.show()