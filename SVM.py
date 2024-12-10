from LoadDataSet import *
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score
def SVM():
    # 加载数据集
    data = LoadData(DataName.BreastCancer)

    # 查看数据集
    print(data)

    # 预处理
    # 特征选择
    x = data.drop('target', axis=1)
    y = data['target']

    # 标准化
    scaler = StandardScaler()
    x = scaler.fit_transform(x)

    # 划分数据集
    x_train, x_test, y_train, y_test = train_test_split(x, y, test_size=0.2, random_state=42)  # 训练模型

    # 训练模型
    svm_model = SVC(kernel='rbf')  # 可根据需要选择合适的核函数
    svm_model.fit(x_train, y_train)

    # 预测
    y_pred = svm_model.predict(x_test)

    # 评估模型
    accuracy = accuracy_score(y_test, y_pred)
    print("准确率：", accuracy)



if __name__ == '__main__':
    pass