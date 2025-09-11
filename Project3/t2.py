import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.feature_selection import f_classif
from xgboost import XGBClassifier
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import LabelBinarizer
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import make_scorer
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import StackingClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import log_loss
from sklearn.preprocessing import StandardScaler
from sklearn.calibration import CalibratedClassifierCV

# 读取数据
X_train = pd.read_csv('X_train.csv')
y_train = pd.read_csv('y_train.csv')
y_train = y_train['label']

# 打印数据维度和前5行
print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)
print(X_train.head())
print(y_train.head())

# 检查缺失值
print("缺失值统计：")
print(X_train.isnull().sum())
print(y_train.isnull().sum())

# 基本统计信息
print(X_train.describe())

class_counts = y_train.value_counts()
print("类别分布：\n", class_counts)
# 可视化类别分布
sns.countplot(x=y_train)
plt.title("Distribution of training set classes")
plt.show()


class FeatureExplorer:
    def __init__(self, X, y):
        self.X = X.select_dtypes(include='number')
        self.y = y
        self.feature_scores = None

    def calculate_feature_importance(self):
        f_scores, p_values = f_classif(self.X, self.y)
        self.feature_scores = pd.DataFrame({
            'feature': self.X.columns,
            'f_score': f_scores,
            'p_value': p_values
        }).sort_values(by='f_score', ascending=False)

    def plot_feature_importance(self, top_k=30):
        if self.feature_scores is None:
            self.calculate_feature_importance()
        plt.figure(figsize=(10, 8))
        sns.barplot(x='f_score', y='feature', data=self.feature_scores.head(top_k), palette='coolwarm', hue=None)
        plt.title('Top Feature Importance (ANOVA F-score)')
        plt.xlabel('F-score')
        plt.ylabel('Feature')
        plt.tight_layout()
        plt.show()

    def plot_feature_correlation(self, top_k=30):
        top_features = self.feature_scores.head(top_k)['feature']
        corr_matrix = self.X[top_features].corr()
        plt.figure(figsize=(12, 10))
        sns.heatmap(corr_matrix, cmap='coolwarm', linewidths=0.5)
        plt.title('Correlation Heatmap of Top Features')
        plt.tight_layout()
        plt.show()

        # Identify highly correlated pairs
        high_corr_pairs = (corr_matrix.abs().where(np.triu(np.ones(corr_matrix.shape), k=1).astype(bool))
                           .stack().sort_values(ascending=False))
        redundant_pairs = high_corr_pairs[high_corr_pairs > 0.9]
        print("Highly correlated feature pairs (correlation > 0.9):")
        print(redundant_pairs)

    def plot_feature_distribution(self, feature, classes_to_plot=[0, 1, 2]):
        plt.figure(figsize=(8, 5))
        for cls in classes_to_plot:
            sns.kdeplot(self.X[feature][self.y == cls], label=f"Class {cls}")
        plt.title(f"Distribution of '{feature}' by Class")
        plt.xlabel(feature)
        plt.legend()
        plt.tight_layout()
        plt.show()

    def recommend_features(self, top_k=30):
        if self.feature_scores is None:
            self.calculate_feature_importance()
        recommended_features = self.feature_scores.head(top_k)['feature'].tolist()
        print(f"Recommended top {top_k} features for modeling:")
        print(recommended_features)
        return recommended_features

explorer = FeatureExplorer(X_train, y_train)

# 计算并画出特征重要性
explorer.plot_feature_importance(top_k=30)

# 画出Top 30特征之间的相关性热力图，查看冗余情况
explorer.plot_feature_correlation(top_k=30)

# 查看单个特征在类别之间的分布（以排名第一特征为例）
top_features = explorer.recommend_features(top_k=5)
for feat in top_features:
    explorer.plot_feature_distribution(feat, classes_to_plot=[0, 1, 2])

# 获得推荐特征用于建模
selected_features = explorer.recommend_features(top_k=30)
X_train_selected = X_train[selected_features]

# 损失
def weighted_log_loss(y_true, y_pred):
    class_counts = np.sum(y_true, axis=0)
    class_weights = 1.0 / class_counts
    class_weights /= np.sum(class_weights)

    sample_weights = np.sum(y_true * class_weights, axis=1)
    loss = -np.mean(sample_weights * np.sum(y_true * np.log(y_pred), axis=1))

    return loss


# 构建 XGBoost 分类器 baseline，注意设置随机种子和关闭标签编码警告
xgb_baseline = XGBClassifier(use_label_encoder=False,
                             eval_metric='mlogloss',
                             random_state=42)

# 使用 StratifiedKFold 进行 5 折交叉验证
cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# 为确保每一折编码一致，提前在全体标签上拟合 LabelBinarizer
lb = LabelBinarizer()
lb.fit(y_train)

#########################################
# 方案1：使用推荐的 30 个特征
#########################################
selected_features = ['48', '17', '263', '270', '172', '283', '231',
                     '90', '111', '88', '249', '87', '71', '141',
                     '99', '218', '46', '86', '182', '234', '230',
                     '26', '265', '216', '5', '41', '157', '299',
                     '63', '292']

X_train_selected = X_train[selected_features]

weighted_losses_selected = []
fold = 1
for train_idx, val_idx in cv.split(X_train_selected, y_train):
    X_tr = X_train_selected.iloc[train_idx]
    X_val = X_train_selected.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    # 训练模型
    xgb_baseline.fit(X_tr, y_tr)
    # 预测验证集概率
    preds_val = xgb_baseline.predict_proba(X_val)
    # 将真实标签转换为 one-hot 格式
    y_val_ohe = lb.transform(y_val)

    loss = weighted_log_loss(y_val_ohe, preds_val)
    weighted_losses_selected.append(loss)
    print(f"Selected Features - Fold {fold} weighted log loss: {loss:.4f}")
    fold += 1

mean_loss_selected = np.mean(weighted_losses_selected)
print(f"\nSelected Features - Baseline weighted cross-entropy loss (5-fold mean): {mean_loss_selected:.4f}")

#########################################
# 方案2：使用完整的 300 个特征
#########################################
X_train_full = X_train  # X_train 已经有300个特征

weighted_losses_full = []
fold = 1
for train_idx, val_idx in cv.split(X_train_full, y_train):
    X_tr = X_train_full.iloc[train_idx]
    X_val = X_train_full.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    # 训练模型
    xgb_baseline.fit(X_tr, y_tr)
    # 预测验证集概率
    preds_val = xgb_baseline.predict_proba(X_val)
    # 使用统一的 LabelBinarizer 转换真实标签
    y_val_ohe = lb.transform(y_val)

    loss = weighted_log_loss(y_val_ohe, preds_val)
    weighted_losses_full.append(loss)
    print(f"Full Features - Fold {fold} weighted log loss: {loss:.4f}")
    fold += 1

mean_loss_full = np.mean(weighted_losses_full)
print(f"\nFull Features - Baseline weighted cross-entropy loss (5-fold mean): {mean_loss_full:.4f}")

# 超参数调优
param_grid = {
    'max_depth': [3, 5, 7],
    'learning_rate': [0.01, 0.05, 0.1],
    'n_estimators': [100, 200, 300],
    'subsample': [0.7, 0.8, 1.0],
    'colsample_bytree': [0.6, 0.7, 1],
    'reg_alpha': [0, 0.1, 0.5],
    'reg_lambda': [1, 1.5, 2]
}

def weighted_log_loss_scorer(estimator, X, y):
    # 预测概率
    y_pred = estimator.predict_proba(X)
    # 使用事先拟合好的 LabelBinarizer 将y转换为one-hot编码
    y_true = lb.transform(y)
    # 返回负的加权交叉熵损失 (因为GridSearchCV是最大化得分)
    return -weighted_log_loss(y_true, y_pred)

grid_search = GridSearchCV(estimator=xgb_baseline,
                           param_grid=param_grid,
                           scoring=weighted_log_loss_scorer,
                           cv=cv,
                           n_jobs=-1,
                           verbose=2)

grid_search.fit(X_train_full, y_train)

print(f"Best parameters: {grid_search.best_params_}")
print(f"Best weighted cross-entropy loss: {grid_search.best_score_:.4f}")

#########################################
# MLP模型实现
#########################################

scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train_full)
mlp_model = MLPClassifier(hidden_layer_sizes=(100,), max_iter=200, random_state=42)

weighted_losses_mlp = []
fold = 1
for train_idx, val_idx in cv.split(X_train_scaled, y_train):
    X_tr = X_train_scaled.iloc[train_idx]
    X_val = X_train_scaled.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    mlp_model.fit(X_tr, y_tr)
    preds_val = mlp_model.predict_proba(X_val)
    y_val_ohe = lb.transform(y_val)

    loss = weighted_log_loss(y_val_ohe, preds_val)
    weighted_losses_mlp.append(loss)
    print(f"MLP Model - Fold {fold} weighted log loss: {loss:.4f}")
    fold += 1

mean_loss_mlp = np.mean(weighted_losses_mlp)
print(f"\nMLP Model - Baseline weighted cross-entropy loss (5-fold mean): {mean_loss_mlp:.4f}")


#########################################
# Stacking集成模型实现
#########################################
stacking_model = StackingClassifier(
    estimators=[('xgb', xgb_baseline), ('mlp', mlp_model)],
    final_estimator=LogisticRegression(max_iter=200),
    cv=cv,
    n_jobs=-1
)

weighted_losses_stacking = []
fold = 1
for train_idx, val_idx in cv.split(X_train_full, y_train):
    X_tr = X_train_full.iloc[train_idx]
    X_val = X_train_full.iloc[val_idx]
    y_tr = y_train.iloc[train_idx]
    y_val = y_train.iloc[val_idx]

    stacking_model.fit(X_tr, y_tr)
    preds_val = stacking_model.predict_proba(X_val)
    y_val_ohe = lb.transform(y_val)

    loss = weighted_log_loss(y_val_ohe, preds_val)
    weighted_losses_stacking.append(loss)
    print(f"Stacking Model - Fold {fold} weighted log loss: {loss:.4f}")
    fold += 1

mean_loss_stacking = np.mean(weighted_losses_stacking)
print(f"\nStacking Model - Baseline weighted cross-entropy loss (5-fold mean): {mean_loss_stacking:.4f}")

# MLP 超参数调优
mlp_param_grid = {
    'hidden_layer_sizes': [(50,), (100,), (100, 50)],
    'alpha': [0.0001, 0.001, 0.01],
    'learning_rate_init': [0.001, 0.01, 0.1]
}

mlp_grid = GridSearchCV(
    estimator=MLPClassifier(max_iter=500, random_state=42),
    param_grid=mlp_param_grid,
    scoring=weighted_log_loss_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=2
)

mlp_grid.fit(X_train_scaled, y_train)

print("MLP最佳参数:", mlp_grid.best_params_)
print("MLP最佳weighted log loss:", mlp_grid.best_score_)

#Stacking（XGBoost + MLP）超参数调优
stacking_param_grid = {
    'final_estimator__C': [0.1, 1, 10]
}

stacking_grid = GridSearchCV(
    estimator=StackingClassifier(
        estimators=[('xgb', xgb_baseline), ('mlp', mlp_grid.best_estimator_)],
        final_estimator=LogisticRegression(max_iter=500),
        cv=cv,
        n_jobs=-1
    ),
    param_grid=stacking_param_grid,
    scoring=weighted_log_loss_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=2
)

stacking_grid.fit(X_train_scaled, y_train)

print("Stacking(XGB+MLP)最佳参数:", stacking_grid.best_params_)
print("Stacking(XGB+MLP)最佳weighted log loss:", stacking_grid.best_score_)

#Stacking(XGB+MLP)2

# 定义基学习器
base_learners = [
    ('xgb', XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)),
    ('mlp', MLPClassifier(max_iter=500, random_state=42))
]

# 定义meta学习器
meta_learner = XGBClassifier(use_label_encoder=False, eval_metric='mlogloss', random_state=42)

# 构建Stacking模型
stacking_model = StackingClassifier(
    estimators=base_learners,
    final_estimator=meta_learner,
    cv=cv,
    n_jobs=-1
)

# 使用CalibratedClassifierCV包裹Stacking整体模型
calibrated_stacking_model = CalibratedClassifierCV(
    base_estimator=stacking_model,
    method='isotonic',
    cv=cv
)
#评估
weighted_losses_calibrated_stacking = []
fold = 1
for train_idx, val_idx in cv.split(X_train_scaled, y_train):
    X_tr, X_val = X_train_scaled[train_idx], X_train_scaled[val_idx]
    y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]

    calibrated_stacking_model.fit(X_tr, y_tr)
    preds_val = calibrated_stacking_model.predict_proba(X_val)
    y_val_ohe = lb.transform(y_val)

    loss = weighted_log_loss(y_val_ohe, preds_val)
    weighted_losses_calibrated_stacking.append(loss)
    print(f"Calibrated Stacking - Fold {fold} weighted log loss: {loss:.4f}")
    fold += 1

mean_loss_calibrated_stacking = np.mean(weighted_losses_calibrated_stacking)
print(f"\nCalibrated Stacking (XGB+MLP base, XGB meta) - weighted cross-entropy loss (5-fold mean): {mean_loss_calibrated_stacking:.4f}")
#超参数调优
param_grid_calibrated_stacking = {
    'base_estimator__xgb__max_depth': [3, 5],
    'base_estimator__xgb__n_estimators': [100, 200],
    'base_estimator__mlp__hidden_layer_sizes': [(50,), (100,)],
    'base_estimator__final_estimator__max_depth': [3, 5],
    'base_estimator__final_estimator__n_estimators': [100, 200]
}

grid_search_calibrated_stacking = GridSearchCV(
    estimator=calibrated_stacking_model,
    param_grid=param_grid_calibrated_stacking,
    scoring=weighted_log_loss_scorer,
    cv=cv,
    n_jobs=-1,
    verbose=2
)

grid_search_calibrated_stacking.fit(X_train_scaled, y_train)

print("最佳参数:", grid_search_calibrated_stacking.best_params_)
print("最佳weighted log loss:", grid_search_calibrated_stacking.best_score_)


# 比较所有模型
results_df = pd.DataFrame({
    'Model': ['MLP', 'Stacking (XGB+MLP)', 'Stacking (XGB+Calibrated)'],
    'Weighted Log Loss': [
        mlp_grid.best_score_,
        stacking_grid.best_score_,
        grid_search_calibrated_stacking.best_score_
    ]
})

print("\n所有模型超参数调优结果对比：")
print(results_df.sort_values(by='Weighted Log Loss', ascending=False))