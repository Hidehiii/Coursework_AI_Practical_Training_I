import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from tqdm import tqdm

# 加载数据集
data = load_breast_cancer()
X, y = data.data, data.target

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 构建流水线：多项式特征 + 随机森林
pipeline = Pipeline([
    ("poly", PolynomialFeatures(include_bias=False)),  # 多项式特征转换
    ("rf", RandomForestRegressor(random_state=42))    # 随机森林
])

# 定义参数网格
param_grid = {
    "poly__degree": [2, 3, 4],  # 增加多项式阶数
    "rf__n_estimators": [100, 200, 300, 500],  # 增加树的数量
    "rf__max_depth": [5, 10, 15, 20],  # 扩展最大深度
    "rf__min_samples_split": [2, 5, 10, 20]  # 增大分裂样本范围
}

# 设置网格搜索
grid_search = GridSearchCV(
    pipeline,
    param_grid,
    scoring="r2",
    cv=5,  # 5折交叉验证
    n_jobs=-1,  # 并行处理
    verbose=2
)

# 执行网格搜索
print("开始网格搜索...")
grid_search.fit(X_train, y_train)
print("网格搜索完成！")

# 打印最佳参数和结果
print("\n最佳参数:", grid_search.best_params_)
print("训练集 R^2:", grid_search.best_score_)

# 使用最佳模型进行测试
best_model = grid_search.best_estimator_
y_train_pred = best_model.predict(X_train)
y_test_pred = best_model.predict(X_test)

# 计算性能指标
train_r2 = r2_score(y_train, y_train_pred)
test_r2 = r2_score(y_test, y_test_pred)
train_mse = mean_squared_error(y_train, y_train_pred)
test_mse = mean_squared_error(y_test, y_test_pred)

print("\n性能指标：")
print("训练集 R^2:", train_r2)
print("测试集 R^2:", test_r2)
print("训练集 MSE:", train_mse)
print("测试集 MSE:", test_mse)

# 保存训练和测试结果
predictions = pd.DataFrame({
    "y_train_true": y_train,
    "y_train_pred": y_train_pred,
    "y_test_true": y_test,
    "y_test_pred": y_test_pred
})
predictions.to_csv("train_test_predictions.csv", index=False)
print("\n训练和测试结果已保存为 'train_test_predictions.csv'")

# 将网格搜索结果保存为 CSV
results_df = pd.DataFrame(grid_search.cv_results_)
results_df.to_csv("random_forest_polynomial_search_results3.csv", index=False)
print("网格搜索结果已保存为 'random_forest_polynomial_search_results3.csv'")
