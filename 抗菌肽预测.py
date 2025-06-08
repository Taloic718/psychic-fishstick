# 集成四个模型
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.ensemble import RandomForestClassifier, VotingClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, f1_score, matthews_corrcoef, roc_auc_score, make_scorer, roc_curve, auc
from imblearn.over_sampling import SMOTE
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import shap
import os
import time
from concurrent.futures import ProcessPoolExecutor
from functools import partial
from tqdm import tqdm  # 进度条显示
from sklearn.metrics import confusion_matrix  # 导入混淆矩阵计算函数

# 设置中文显示
plt.rcParams["font.family"] = ["SimHei", "WenQuanYi Micro Hei", "Heiti TC"]
plt.rcParams['axes.unicode_minus'] = False  # 解决负号显示问题

# 数据文件路径
positive_dir = r"C:\Users\Lenovo\Desktop\course design\correct"
negative_dir = r"C:\Users\Lenovo\Desktop\course design\false"
pos_file_path = os.path.join(positive_dir, '正DDE.csv')
neg_file_path = os.path.join(negative_dir, '负DDE.csv')

# 配置参数
RANDOM_SEED = 42  # 随机种子，确保可复现性
CV_FOLDS = 5  # 交叉验证折数
N_JOBS = -1  # 并行任务数（-1表示使用所有CPU核心）
MODEL_PARAMS = {
    "XGBoost": {
        "n_estimators": 150,
        "learning_rate": 0.1,
        "max_depth": 6,
        "min_child_weight": 1,
        "gamma": 0,
        "subsample": 0.8,
        "colsample_bytree": 0.8,
        "reg_alpha": 0.1,
        "reg_lambda": 0.1,
        "use_label_encoder": False,
        "eval_metric": "logloss",
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
    },
    "Random Forest": {
        "n_estimators": 150,
        "max_depth": 10,
        "min_samples_split": 5,
        "min_samples_leaf": 2,
        "max_features": "sqrt",
        "bootstrap": True,
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
    },
    "Logistic Regression": {
        "C": 1.0,
        "penalty": "elasticnet",
        "solver": "saga",
        "l1_ratio": 0.5,
        "max_iter": 1000,
        "random_state": RANDOM_SEED,
        "n_jobs": N_JOBS,
    },
    "SVM": {
        "kernel": 'linear',
        "C": 1.0,
        "probability": True,
        "cache_size": 500
    }
}


# 数据加载与预处理
def load_and_preprocess_data():
    """加载并预处理数据"""
    print("开始数据预处理...")
    start_time = time.time()

    # 1. 数据准备
    try:
        pos_df = pd.read_csv(pos_file_path, header=None)
    except FileNotFoundError:
        print(f"正样本文件 {pos_file_path} 不存在，请检查路径。")
        raise
    pos_df[0] = 1  # 设置正样本标签为1

    try:
        neg_df = pd.read_csv(neg_file_path, header=None)
    except FileNotFoundError:
        print(f"负样本文件 {neg_file_path} 不存在，请检查路径。")
        raise
    neg_df[0] = 0  # 设置负样本标签为0

    # 合并数据集
    df = pd.concat([pos_df, neg_df], ignore_index=True)

    # 2. 数据预处理
    X = df.iloc[:, 1:].values
    y = df.iloc[:, 0].values

    # 处理样本不平衡
    smote = SMOTE(random_state=42, k_neighbors=7)
    X_res, y_res = smote.fit_resample(X, y)

    # 划分数据集
    X_train, X_test, y_train, y_test = train_test_split(
        X_res, y_res, test_size=0.2, random_state=42, stratify=y_res
    )

    print(f"数据预处理完成，耗时: {time.time() - start_time:.2f}秒")
    return X_res, y_res, X_train, X_test, y_train, y_test


# 模型评估函数
def evaluate_model(model, model_name, X_res, y_res, X_train, X_test, y_train, y_test, cv_folds=5):
    """评估单个模型的性能（添加 X_res 和 y_res 参数）"""
    print(f"\n===== 开始评估模型: {model_name} =====")
    start_time = time.time()

    # 训练模型
    model.fit(X_train, y_train)

    # 评估训练集性能
    y_train_pred = model.predict(X_train)
    train_metrics = {
        "Accuracy": accuracy_score(y_train, y_train_pred),
        "F1-Score": f1_score(y_train, y_train_pred),
        "MCC": matthews_corrcoef(y_train, y_train_pred),
    }

    # 计算训练集的混淆矩阵和灵敏度、特异度
    tn, fp, fn, tp = confusion_matrix(y_train, y_train_pred).ravel()
    train_metrics["Sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    train_metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 获取训练集预测概率（用于ROC曲线）
    y_train_prob = None
    if hasattr(model, 'predict_proba'):
        train_metrics["AUC-ROC"] = roc_auc_score(y_train, model.predict_proba(X_train)[:, 1])
        y_train_prob = model.predict_proba(X_train)[:, 1]
    elif hasattr(model, 'decision_function'):
        train_metrics["AUC-ROC"] = roc_auc_score(y_train, model.decision_function(X_train))
        y_train_prob = model.decision_function(X_train)

    print("\n训练集性能:")
    for metric, value in train_metrics.items():
        print(f"{metric}: {value:.4f}")

    # 评估测试集性能
    y_test_pred = model.predict(X_test)
    test_metrics = {
        "Accuracy": accuracy_score(y_test, y_test_pred),
        "F1-Score": f1_score(y_test, y_test_pred),
        "MCC": matthews_corrcoef(y_test, y_test_pred),
    }

    # 计算测试集的混淆矩阵和灵敏度、特异度
    tn, fp, fn, tp = confusion_matrix(y_test, y_test_pred).ravel()
    test_metrics["Sensitivity"] = tp / (tp + fn) if (tp + fn) > 0 else 0
    test_metrics["Specificity"] = tn / (tn + fp) if (tn + fp) > 0 else 0

    # 获取测试集预测概率（用于ROC曲线）
    y_test_prob = None
    if hasattr(model, 'predict_proba'):
        test_metrics["AUC-ROC"] = roc_auc_score(y_test, model.predict_proba(X_test)[:, 1])
        y_test_prob = model.predict_proba(X_test)[:, 1]
    elif hasattr(model, 'decision_function'):
        test_metrics["AUC-ROC"] = roc_auc_score(y_test, model.decision_function(X_test))
        y_test_prob = model.decision_function(X_test)

    print("\n测试集性能:")
    for metric, value in test_metrics.items():
        print(f"{metric}: {value:.4f}")

    # 5折交叉验证（使用传入的 X_res 和 y_res）
    print(f"\n{cv_folds}折交叉验证结果:")
    cv_metrics = {}

    # 自定义评估函数
    scorers = {
        'accuracy': make_scorer(accuracy_score),
        'f1': make_scorer(f1_score),
        'mcc': make_scorer(matthews_corrcoef),
        'roc_auc': make_scorer(roc_auc_score, needs_proba=True),
        'sensitivity': make_scorer(lambda y_true, y_pred: confusion_matrix(y_true, y_pred).ravel()[3] /
                                                          (confusion_matrix(y_true, y_pred).ravel()[3] +
                                                           confusion_matrix(y_true, y_pred).ravel()[2])
        if (confusion_matrix(y_true, y_pred).ravel()[3] + confusion_matrix(y_true, y_pred).ravel()[2]) > 0 else 0),
        'specificity': make_scorer(lambda y_true, y_pred: confusion_matrix(y_true, y_pred).ravel()[0] /
                                                          (confusion_matrix(y_true, y_pred).ravel()[0] +
                                                           confusion_matrix(y_true, y_pred).ravel()[1])
        if (confusion_matrix(y_true, y_pred).ravel()[0] + confusion_matrix(y_true, y_pred).ravel()[1]) > 0 else 0)
    }

    for metric_name, scorer in scorers.items():
        if metric_name == 'roc_auc' and not hasattr(model, 'predict_proba'):
            continue

        cv_scores = cross_val_score(
            model, X_res, y_res,  # 使用传入的过采样数据集
            cv=StratifiedKFold(n_splits=cv_folds, shuffle=True, random_state=42),
            scoring=scorer,
            n_jobs=-1  # 使用所有CPU核心进行交叉验证
        )

        cv_metrics[metric_name] = cv_scores
        print(f"{metric_name}: {np.mean(cv_scores):.4f} ± {np.std(cv_scores):.4f}")

    # 特征重要性（随机森林和XGBoost）
    feature_importance = None
    if model_name in ["Random Forest", "XGBoost"]:
        importances = model.feature_importances_
        indices = np.argsort(importances)[::-1]

        plt.figure(figsize=(12, 6))
        plt.title(f"Top 20 Important DDE Features - {model_name}")
        plt.bar(range(20), importances[indices[:20]], align='center')
        plt.xticks(range(20), indices[:20], rotation=90)
        plt.tight_layout()
        plt.savefig(f"{model_name}_feature_importance.png")
        plt.close()  # 关闭图形以节省内存

        feature_importance = importances

    print(f"模型 {model_name} 评估完成，耗时: {time.time() - start_time:.2f}秒")
    return {
        'model': model,
        'test_metrics': test_metrics,
        'cv_metrics': cv_metrics,
        'feature_importance': feature_importance,
        'y_test_prob': y_test_prob  # 保存测试集预测概率用于ROC曲线
    }


# 绘制ROC曲线函数
def plot_roc_curves(models_dict, X_test, y_test, ensemble_model=None, ensemble_name="Ensemble(DDE)"):
    """绘制多个模型的ROC曲线"""
    plt.figure(figsize=(10, 8))

    # 绘制每个基模型的ROC曲线
    for name, model_result in models_dict.items():
        y_test_prob = model_result['y_test_prob']
        if y_test_prob is not None:
            fpr, tpr, _ = roc_curve(y_test, y_test_prob)
            roc_auc = auc(fpr, tpr)
            plt.plot(fpr, tpr, lw=2, label=f'{name} (AUC = {roc_auc:.3f})')

    # 绘制集成模型的ROC曲线
    if ensemble_model is not None:
        if hasattr(ensemble_model, 'predict_proba'):
            y_test_prob_ensemble = ensemble_model.predict_proba(X_test)[:, 1]
        elif hasattr(ensemble_model, 'decision_function'):
            y_test_prob_ensemble = ensemble_model.decision_function(X_test)
        else:
            print("集成模型无法获取预测概率或决策函数值，无法绘制ROC曲线")
            return

        fpr_ensemble, tpr_ensemble, _ = roc_curve(y_test, y_test_prob_ensemble)
        roc_auc_ensemble = auc(fpr_ensemble, tpr_ensemble)
        plt.plot(fpr_ensemble, tpr_ensemble, lw=3, linestyle='--',
                 label=f'{ensemble_name} (AUC = {roc_auc_ensemble:.3f})')

    # 绘制随机猜测线
    plt.plot([0, 1], [0, 1], 'k--', lw=2)

    # 设置图表属性
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('假阳性率 (1-特异性)', fontsize=12)
    plt.ylabel('真阳性率 (灵敏度)', fontsize=12)
    plt.title('不同模型在测试集上的ROC曲线比较', fontsize=14)
    plt.legend(loc="lower right", fontsize=10)
    plt.grid(alpha=0.3)

    # 保存图表
    plt.tight_layout()
    plt.savefig('ROC_curves_comparison.png')
    plt.close()
    print("ROC曲线已保存至 ROC_curves_comparison.png")


if __name__ == "__main__":
    # 加载和预处理数据
    X_res, y_res, X_train, X_test, y_train, y_test = load_and_preprocess_data()

    # 模型构建
    models = {
        "XGBoost": XGBClassifier(**MODEL_PARAMS["XGBoost"]),
        "Random Forest": RandomForestClassifier(**MODEL_PARAMS["Random Forest"]),
        "Logistic Regression": LogisticRegression(**MODEL_PARAMS["Logistic Regression"]),
        "SVM": SVC(**MODEL_PARAMS["SVM"])
    }

    # 使用进程池并行评估模型
    print("\n开始并行训练和评估模型...")
    start_time = time.time()

    with ProcessPoolExecutor(max_workers=len(models)) as executor:
        # 部分应用参数，显式传递 X_res 和 y_res
        eval_func = partial(
            evaluate_model,
            X_res=X_res,
            y_res=y_res,
            X_train=X_train,
            X_test=X_test,
            y_train=y_train,
            y_test=y_test,
            cv_folds=CV_FOLDS
        )

        # 提交所有模型进行并行评估
        futures = {name: executor.submit(eval_func, model, name) for name, model in models.items()}

        # 收集结果
        results = {}
        for name, future in tqdm(futures.items(), desc="模型训练进度", total=len(models)):
            results[name] = future.result()

    print(f"\n所有模型评估完成，总耗时: {time.time() - start_time:.2f}秒")

    # 结果汇总
    print("\n===== 模型性能比较 =====")
    test_results = {name: result['test_metrics'] for name, result in results.items()}
    results_df = pd.DataFrame(test_results).T
    print(results_df)
    results_df.to_csv('model_performance_full.csv', index=True)
    print("模型性能结果已保存至 model_performance_full.csv 文件，可在表格软件中查看完整内容")

    # 选择四个较优模型
    sorted_models = results_df.sort_values(by='AUC-ROC', ascending=False).index[:4].tolist()
    top_models = [(name, results[name]['model']) for name in sorted_models]

    # 构建集成模型
    voting_clf = VotingClassifier(
        estimators=top_models,
        voting="soft",  # 概率加权
        n_jobs=N_JOBS
    )

    # 训练集成模型
    voting_clf.fit(X_train, y_train)
    y_pred = voting_clf.predict(X_test)

    # 评估集成模型
    y_pred = voting_clf.predict(X_test)

    # 计算混淆矩阵
    tn, fp, fn, tp = confusion_matrix(y_test, y_pred).ravel()

    ensemble_metrics = {
        "Accuracy": accuracy_score(y_test, y_pred),
        "F1-Score": f1_score(y_test, y_pred),
        "MCC": matthews_corrcoef(y_test, y_pred),
        "Sensitivity": tp / (tp + fn) if (tp + fn) > 0 else 0,
        "Specificity": tn / (tn + fp) if (tn + fp) > 0 else 0,
    }
    if hasattr(voting_clf, "predict_proba"):
        ensemble_metrics["AUC-ROC"] = roc_auc_score(y_test, voting_clf.predict_proba(X_test)[:, 1])

    print("\n=== 集成模型性能 ===")
    for metric, value in ensemble_metrics.items():
        print(f"{metric}: {value:.4f}")

    # 绘制ROC曲线
    plot_roc_curves(results, X_test, y_test, voting_clf)

    # 保存集成模型
    joblib.dump(voting_clf, "ensemble_antimicrobial_peptide_model.pkl")
    print("集成抗菌肽预测模型已保存至 ensemble_antimicrobial_peptide_model.pkl")