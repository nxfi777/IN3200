import pandas as pd
import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix
from sklearn.model_selection import cross_val_score
import warnings
warnings.filterwarnings('ignore')

def load_data(PATH="UCI HAR Dataset/"):
    features_df = pd.read_csv(PATH + "features.txt", sep="\\s+", header=None, names=["idx", "feature"])
    feature_names = []
    seen_features = {}
    
    for feature in features_df["feature"]:
        if feature in seen_features:
            seen_features[feature] += 1
            feature_names.append(f"{feature}_{seen_features[feature]}")
        else:
            seen_features[feature] = 0
            feature_names.append(feature)
    
    activity_labels_df = pd.read_csv(PATH + "activity_labels.txt", sep="\\s+", header=None, names=["id", "activity"])
    activity_map = dict(zip(activity_labels_df.id, activity_labels_df.activity))
    
    X_train = pd.read_csv(PATH + "train/X_train.txt", sep="\\s+", header=None, names=feature_names)
    y_train = pd.read_csv(PATH + "train/y_train.txt", sep="\\s+", header=None, names=["Activity"])
    X_test = pd.read_csv(PATH + "test/X_test.txt", sep="\\s+", header=None, names=feature_names)
    y_test = pd.read_csv(PATH + "test/y_test.txt", sep="\\s+", header=None, names=["Activity"])
    
    subject_train = pd.read_csv(PATH + "train/subject_train.txt", header=None, names=["Subject"])
    subject_test = pd.read_csv(PATH + "test/subject_test.txt", header=None, names=["Subject"])
    
    y_train["Activity"] = y_train["Activity"].map(activity_map)
    y_test["Activity"] = y_test["Activity"].map(activity_map)
    
    return X_train, X_test, y_train, y_test, subject_train, subject_test

def to_binary_label(activity):
    return 1 if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"] else 0

def objective(trial, X_train, y_train):
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
    C = trial.suggest_float('C', 1e-3, 1e3, log=True)
    
    params = {'C': C, 'kernel': kernel, 'class_weight': 'balanced', 'random_state': 42}
    
    if kernel in ['poly', 'rbf']:
        params['gamma'] = trial.suggest_float('gamma', 1e-4, 1e1, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 4)
    
    use_pca = trial.suggest_categorical('use_pca', [True, False])
    pipeline_steps = [('scaler', StandardScaler())]
    
    if use_pca:
        pca_method = trial.suggest_categorical('pca_method', ['n_components', 'variance_ratio'])
        if pca_method == 'n_components':
            max_components = X_train.shape[1]
            min_components = max(int(0.1 * max_components), 2)
            n_components = trial.suggest_int('n_components', min_components, max_components)
            pipeline_steps.append(('pca', PCA(n_components=n_components)))
        else:
            variance_ratio = trial.suggest_float('variance_ratio', 0.8, 0.999)
            pipeline_steps.append(('pca', PCA(n_components=variance_ratio)))
    
    pipeline_steps.append(('svc', SVC(**params)))
    pipeline = Pipeline(pipeline_steps)
    
    try:
        return cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=-1).mean()
    except:
        return float('-inf')

def main():
    X_train, X_test, y_train, y_test, subject_train, subject_test = load_data()
    
    print(f"Data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    y_train_binary = y_train["Activity"].apply(to_binary_label)
    y_test_binary = y_test["Activity"].apply(to_binary_label)
    
    print("\nClass distribution:")
    print(y_train_binary.value_counts(normalize=True))
    
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train_binary), n_trials=30, n_jobs=-1)
    
    best_params = study.best_params
    print(f"\nBest params: {best_params}")
    print(f"Best CV accuracy: {study.best_value:.4f}")
    
    final_params = {
        'C': best_params['C'],
        'kernel': best_params['kernel'],
        'class_weight': 'balanced',
        'random_state': 42
    }
    
    if best_params['kernel'] in ['poly', 'rbf']:
        final_params['gamma'] = best_params['gamma']
    if best_params['kernel'] == 'poly':
        final_params['degree'] = best_params['degree']
    
    final_pipeline_steps = [('scaler', StandardScaler())]
    
    if best_params.get('use_pca', False):
        if best_params.get('pca_method') == 'n_components':
            final_pipeline_steps.append(('pca', PCA(n_components=best_params['n_components'])))
        else:
            final_pipeline_steps.append(('pca', PCA(n_components=best_params['variance_ratio'])))
    
    final_pipeline_steps.append(('svc', SVC(**final_params)))
    final_pipeline = Pipeline(final_pipeline_steps)
    
    final_pipeline.fit(X_train, y_train_binary)
    y_pred = final_pipeline.predict(X_test)
    
    print("\nTest Results:")
    print(confusion_matrix(y_test_binary, y_pred))
    print("\nClassification Report:")
    print(classification_report(y_test_binary, y_pred))
    
    if best_params.get('use_pca', False):
        pca = final_pipeline.named_steps.get('pca')
        if pca:
            print(f"\nPCA: {pca.n_components_} components, {np.sum(pca.explained_variance_ratio_):.4f} variance")
    
    if best_params['kernel'] == 'linear':
        svc = final_pipeline.named_steps['svc']
        feature_importance = np.abs(svc.coef_[0] @ pca.components_) if 'pca' in final_pipeline.named_steps else np.abs(svc.coef_[0])
        top_features = np.argsort(feature_importance)[-10:][::-1]
        print("\nTop 10 Features:")
        for idx in top_features:
            print(f"{idx}: {feature_importance[idx]:.4f}")

if __name__ == "__main__":
    main() 