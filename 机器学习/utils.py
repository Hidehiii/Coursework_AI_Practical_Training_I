import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_breast_cancer
import matplotlib.pyplot as plt
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
import numpy as np
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LinearRegression, Ridge, Lasso
from sklearn.svm import SVR
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import RandomForestRegressor, GradientBoostingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.kernel_ridge import KernelRidge
from scipy.signal import savgol_filter

plt.rcParams['font.sans-serif'] = ['STHeiti']  # 指定默认字体
plt.rcParams['axes.unicode_minus'] = False  # 解决保存图像是负号'-'显示为方块的问题


def plot_boxplot(df, feature, target='target'):
    """绘制指定特征的箱线图，按目标分类"""
    plt.figure(figsize=(8, 6))
    sns.boxplot(x=target, y=feature, data=df)
    plt.title(f'{feature} 的箱线图')
    plt.xlabel('类别')
    plt.ylabel(feature)
    plt.show()

def load_breast_cancer_data():
    """加载威斯康辛州乳腺癌数据集并返回DataFrame格式数据和目标名称。"""
    data = load_breast_cancer()
    df = pd.DataFrame(data.data, columns=data.feature_names)
    df['target'] = data.target
    return df, data.target_names

def summarize_data(df):
    """打印数据的基本信息和统计摘要。"""
    print("数据基本信息：")
    print(df.info())
    print("\n统计摘要：")
    print(df.describe())
    print("\n目标变量分布：")
    print(df['target'].value_counts())

def plot_target_distribution(df, target_names):
    """绘制目标变量分布。"""
    sns.countplot(x='target', data=df)
    plt.xticks([0, 1], target_names)
    plt.title("目标变量分布")
    plt.xlabel("类别")
    plt.ylabel("样本数")
    plt.show()

def plot_feature_distributions(df, features, bins=30):
    """绘制给定特征的分布图。"""
    for feature in features:
        plt.figure(figsize=(6, 4))
        sns.histplot(df[feature], kde=True, bins=bins)
        plt.title(f"{feature} 分布")
        plt.xlabel(feature)
        plt.ylabel("频率")
        plt.show()

def plot_correlation_matrix(df):
    """绘制相关性矩阵的热力图。"""
    corr_matrix = df.corr()
    plt.figure(figsize=(12, 10))
    sns.heatmap(corr_matrix, annot=False, cmap='coolwarm', linewidths=0.5)
    plt.title("特征相关性矩阵")
    plt.show()

def analyze_top_correlations(df, target_column='target', top_n=5):
    """计算与目标变量最相关的特征。"""
    correlations = df.corr()[target_column].sort_values(ascending=False)
    top_correlations = correlations.head(top_n + 1)  # 包括目标列自身
    bottom_correlations = correlations.tail(top_n)
    print("与目标变量最正相关的特征：")
    print(top_correlations)
    print("\n与目标变量最负相关的特征：")
    print(bottom_correlations)
    return top_correlations, bottom_correlations

def plot_pairwise_relationships(df, features, target='target'):
    """绘制多特征的成对关系图"""
    sns.pairplot(df[features + [target]], hue=target, palette='coolwarm')
    plt.show()

def calculate_feature_statistics(df, target='target'):
    """按目标类别计算特征的均值和标准差"""
    stats = df.groupby(target).agg(['mean', 'std'])
    print("按类别计算的特征统计：")
    print(stats)
    return stats

def plot_grouped_histograms(df, features, target='target', bins=30):
    """按目标类别绘制多个特征的分组直方图"""
    for feature in features:
        plt.figure(figsize=(8, 6))
        for label in df[target].unique():
            subset = df[df[target] == label]
            sns.histplot(subset[feature], kde=True, bins=bins, label=f'{target}={label}')
        plt.title(f'{feature} 的分组直方图')
        plt.xlabel(feature)
        plt.ylabel('频率')
        plt.legend()
        plt.show()


def standardize_data(df, features):
    """对指定特征进行标准化"""
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(df[features])
    standardized_df = pd.DataFrame(standardized_data, columns=features)
    print("标准化后的数据：")
    print(standardized_df.describe())
    return standardized_df

# ========================
# 数据预处理方法
# ========================
def apply_snv(data):
    """标准正态变换 (SNV)"""
    return (data - np.mean(data, axis=0)) / np.std(data, axis=0)

def apply_sg(data, window_size=9, polyorder=3):
    """Savitzky-Golay 平滑滤波 (SG)"""
    # 增加窗口大小和多项式阶数，以更好平滑数据
    return savgol_filter(data, window_length=window_size, polyorder=polyorder, axis=0)

def apply_pca(data, n_components=8):
    """主成分分析 (PCA) 降维"""
    # 增加主成分数，以保留更多特征
    pca = PCA(n_components=n_components)
    reduced_data = pca.fit_transform(data)
    return reduced_data, pca

def apply_derivative(data, order=2):
    """计算导数 (DT)"""
    # 使用二阶导数，提取更高阶变化特征
    return np.diff(data, n=order, axis=0)

def apply_standardization(data):
    """标准化数据"""
    scaler = StandardScaler()
    standardized_data = scaler.fit_transform(data)
    return standardized_data, scaler

def apply_minmax_scaling(data):
    """最小最大归一化"""
    min_val = np.min(data, axis=0)
    max_val = np.max(data, axis=0)
    return (data - min_val) / (max_val - min_val)

def apply_log_transformation(data):
    """对数据应用对数变换（避免负值）"""
    return np.log1p(data)

def apply_polynomial_features(data, degree=3):
    """生成多项式特征"""
    from sklearn.preprocessing import PolynomialFeatures
    # 增加多项式阶数至3，适应乳腺癌数据的复杂性
    poly = PolynomialFeatures(degree=degree, include_bias=False)
    return poly.fit_transform(data)

# ========================
# 模型定义
# ========================
def create_models():
    """创建一组常用的回归模型"""
    models = {
        "LinearRegression": LinearRegression(),  # 无需设置 max_iter，线性回归直接计算解析解
        "Ridge": Ridge(alpha=2.0, max_iter=10000),  # 增大 alpha 值，进一步减少过拟合
        "Lasso": Lasso(alpha=0.01, max_iter=10000),  # 调整 alpha，提升稀疏性表现
        "SVM_RBF": SVR(kernel="rbf", C=10, gamma=0.1, max_iter=2000),  # 提高 gamma 的值，增强模型表达能力
        "SVM_Linear": SVR(kernel="linear", C=1.0, max_iter=2000),  # 增大 C，提高拟合能力
        "DecisionTree": DecisionTreeRegressor(max_depth=10, min_samples_split=5),  # 增大最大深度，限制最小样本分裂
        "RandomForest": RandomForestRegressor(
            n_estimators=500, max_depth=12, min_samples_split=5, random_state=42
        ),  # 增加树的数量，提高深度，限制最小样本分裂
        "GradientBoosting": GradientBoostingRegressor(
            n_estimators=200, learning_rate=0.05, max_depth=6, random_state=42
        ),  # 增加基学习器数量，适度增加深度
        "MLP": MLPRegressor(
            hidden_layer_sizes=(100, 50), max_iter=3000, alpha=0.0001, random_state=42
        ),  # 增加隐藏层的复杂度
        "KernelRidge": KernelRidge(kernel="rbf", alpha=0.1, gamma=0.1),  # 使用 RBF 核，调小 alpha 和 gamma 提高适配能力
    }
    return models

# ========================
# 示例调用接口
# ========================
def get_preprocessing_methods():
    """返回所有可用的预处理方法"""
    return {
        "None": None,  # 不进行预处理
        "SNV": apply_snv,  # 标准正态变换
        "SG": lambda data: apply_sg(data, window_size=9, polyorder=3),  # 平滑滤波
        "PCA": lambda data: apply_pca(data, n_components=8)[0],  # PCA 降维
        "Derivative": lambda data: apply_derivative(data, order=2),  # 二阶导数
        "Standardization": lambda data: apply_standardization(data)[0],  # 标准化
        "MinMaxScaling": apply_minmax_scaling,  # 最小最大归一化
        "LogTransformation": apply_log_transformation,  # 对数变换
        "PolynomialFeatures": lambda data: apply_polynomial_features(data, degree=3),  # 多项式特征生成
    }

