from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.datasets import load_breast_cancer

# 加载数据
data = load_breast_cancer()
X, y = data.data, data.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 参数组
param_sets = [
    {'poly__degree': 3, 'rf__max_depth': 5, 'rf__min_samples_split': 2, 'rf__n_estimators': 500},
    {'poly__degree': 2, 'rf__max_depth': 5, 'rf__min_samples_split': 5, 'rf__n_estimators': 100},
    {'poly__degree': 3, 'rf__max_depth': 5, 'rf__min_samples_split': 2, 'rf__n_estimators': 300}
]

# 遍历参数组
for params in param_sets:
    # 构建流水线
    pipeline = Pipeline([
        ("poly", PolynomialFeatures(degree=params["poly__degree"], include_bias=False)),
        ("rf", RandomForestRegressor(
            n_estimators=params["rf__n_estimators"],
            max_depth=params["rf__max_depth"],
            min_samples_split=params["rf__min_samples_split"],
            random_state=42
        ))
    ])

    # 训练模型
    pipeline.fit(X_train, y_train)

    # 预测结果
    y_train_pred = pipeline.predict(X_train)
    y_test_pred = pipeline.predict(X_test)

    # 计算指标
    train_r2 = r2_score(y_train, y_train_pred)
    test_r2 = r2_score(y_test, y_test_pred)
    train_mse = mean_squared_error(y_train, y_train_pred)
    test_mse = mean_squared_error(y_test, y_test_pred)

    print(f"参数组: {params}")
    print("训练集 R^2:", train_r2)
    print("测试集 R^2:", test_r2)
    print("训练集 MSE:", train_mse)
    print("测试集 MSE:", test_mse)
    print("-" * 50)
