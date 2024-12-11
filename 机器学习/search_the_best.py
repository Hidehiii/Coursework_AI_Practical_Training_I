import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_breast_cancer
from sklearn.pipeline import Pipeline
from tqdm import tqdm
from sklearn.ensemble import RandomForestRegressor
from sklearn.preprocessing import PolynomialFeatures

# 加载威斯康辛乳腺癌数据
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
    "poly__degree": [2, 3],  # 多项式阶数
    "rf__n_estimators": [100, 200],  # 树的数量
    "rf__max_depth": [5, 10],  # 树的最大深度
    "rf__min_samples_split": [2, 5]  # 分裂所需最小样本数
}

# 初始化总进度条
param_combinations = len(param_grid["poly__degree"]) * len(param_grid["rf__n_estimators"]) * \
                     len(param_grid["rf__max_depth"]) * len(param_grid["rf__min_samples_split"])
progress = tqdm(total=param_combinations, desc="总进度", unit="步")

# 初始化结果存储列表
results = []

# 手动网格搜索实现进度条更新
for degree in param_grid["poly__degree"]:
    for n_estimators in param_grid["rf__n_estimators"]:
        for max_depth in param_grid["rf__max_depth"]:
            for min_samples_split in param_grid["rf__min_samples_split"]:
                # 设置流水线参数
                pipeline.set_params(
                    poly__degree=degree,
                    rf__n_estimators=n_estimators,
                    rf__max_depth=max_depth,
                    rf__min_samples_split=min_samples_split
                )

                try:
                    # 训练流水线
                    pipeline.fit(X_train, y_train)

                    # 模型评估
                    y_test_pred = pipeline.predict(X_test)
                    Rp = r2_score(y_test, y_test_pred)
                    MSEp = mean_squared_error(y_test, y_test_pred)

                    # 记录结果
                    result = {
                        "degree": degree,
                        "n_estimators": n_estimators,
                        "max_depth": max_depth,
                        "min_samples_split": min_samples_split,
                        "Rp": Rp,
                        "MSEp": MSEp
                    }
                    results.append(result)
                except Exception as e:
                    print(f"组合 degree={degree}, n_estimators={n_estimators}, "
                          f"max_depth={max_depth}, min_samples_split={min_samples_split} 运行失败，错误信息: {e}")

                # 更新总进度条
                progress.update(1)

# 关闭总进度条
progress.close()

# 转换结果为 DataFrame 并保存
results_df = pd.DataFrame(results)
results_df.to_csv("random_forest_poly_search_results.csv", index=False)

# 打印最佳结果
best_result = results_df.sort_values(by="Rp", ascending=False).head(1)
print("\n最佳结果:")
print(best_result)
