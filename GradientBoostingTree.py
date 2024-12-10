from sklearn.model_selection import train_test_split
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.metrics import accuracy_score
from sklearn.preprocessing import StandardScaler
from LoadDataSet import *

def GradientBoostingTree():
    # 加载数据集
    data = LoadData(DataName.BreastCancer)

    # 查看数据集
    print(data)

    # 预处理
    # 特征选择
    x = data.data
    y = data.target

    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)

    # 训练梯度提升树模型
    gbdt_model = GradientBoostingClassifier(n_estimators=100, random_state=42)  # 可调整n_estimators参数
    gbdt_model.fit(x_train, y_train)

    # 预测
    y_pred = gbdt_model.predict(x_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print("Gradient Boosting Tree准确率：", accuracy)


if __name__ == '__main__':
    pass