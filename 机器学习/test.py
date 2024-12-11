import utils
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.datasets import load_breast_cancer
from tqdm import tqdm

# 加载威斯康辛乳腺癌数据
data = load_breast_cancer()
X, y = data.data, data.target

# 获取预处理方法和模型
preprocessing_methods = utils.get_preprocessing_methods()
models = utils.create_models()

# 划分数据集
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# 初始化总进度条
total_combinations = len(preprocessing_methods) * len(models)
total_progress = tqdm(total=total_combinations, desc="总进度", unit="组合", position=0)

# 定义输出文件路径
output_path = "model_performance_results2.csv"

# 初始化结果存储列表
results = []

# 遍历预处理和模型组合
for pre_name, pre_func in preprocessing_methods.items():
    for model_name, model in models.items():
        # 获取模型的最大迭代次数
        max_iter = getattr(model, "max_iter", 1)  # 如果模型没有 max_iter 参数，则设为 1
        current_progress = tqdm(
            total=max_iter,  # 使用最大迭代次数初始化进度条
            desc=f"正在运行: 预处理: {pre_name}, 模型: {model_name}",
            unit="it",  # 单位为迭代
            position=1,
            leave=False,
        )

        try:
            # 数据预处理
            if pre_func:
                if pre_name == "PCA":
                    X_train_pre, pca_model = utils.apply_pca(X_train, n_components=5)  # PCA 处理
                    X_test_pre = pca_model.transform(X_test)
                elif pre_name == "Derivative":
                    X_train_pre = pre_func(X_train)
                    X_test_pre = pre_func(X_test)
                    y_train_pre = y_train[:len(X_train_pre)]
                    y_test_pre = y_test[:len(X_test_pre)]
                else:
                    X_train_pre = pre_func(X_train)
                    X_test_pre = pre_func(X_test)
                    y_train_pre, y_test_pre = y_train, y_test
            else:
                X_train_pre, X_test_pre = X_train, X_test
                y_train_pre, y_test_pre = y_train, y_test

            # 模型训练与预测
            for iteration in range(max_iter):
                # 模拟迭代更新进度
                # Note: 部分模型（如线性回归）并无逐步迭代信息，可直接 fit 整体数据。
                if hasattr(model, "partial_fit"):  # 如果模型支持增量训练
                    model.partial_fit(X_train_pre, y_train_pre)
                else:
                    model.fit(X_train_pre, y_train_pre)
                    break  # 不支持迭代的模型只更新一次
                current_progress.update(1)  # 更新当前进度条

            y_train_pred = model.predict(X_train_pre)
            y_test_pred = model.predict(X_test_pre)

            # 计算指标
            Rc = r2_score(y_train_pre, y_train_pred)
            Rp = r2_score(y_test_pre, y_test_pred)
            MSEc = mean_squared_error(y_train_pre, y_train_pred)
            MSEp = mean_squared_error(y_test_pre, y_test_pred)

            # 记录结果
            result = {
                "Preprocessing": pre_name,
                "Model": model_name,
                "Rc": Rc,
                "Rp": Rp,
                "MSEc": MSEc,
                "MSEp": MSEp
            }
            results.append(result)

            # 写入 CSV 文件（追加模式）
            pd.DataFrame([result]).to_csv(output_path, mode='a', header=not pd.io.common.file_exists(output_path), index=False)

        except Exception as e:
            print(f"组合 {pre_name} + {model_name} 运行失败，错误信息: {e}")

        # 更新总进度条
        current_progress.close()
        total_progress.update(1)  # 总进度条只更新模型完成情况

# 关闭总进度条
total_progress.close()

# 打印最终分析结果
print("\n分析完成！最佳组合结果如下：")
try:
    results_df = pd.read_csv(output_path)
    best_result = results_df.sort_values(by="Rp", ascending=False).head(1)
    print(best_result)
except Exception as e:
    print(f"读取结果文件失败: {e}")
