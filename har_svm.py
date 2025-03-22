import pandas as pd
import numpy as np
import optuna
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.svm import SVC
from sklearn.metrics import classification_report, confusion_matrix, roc_curve, auc
from sklearn.model_selection import cross_val_score, learning_curve
import matplotlib.pyplot as plt
import seaborn as sns
import warnings
warnings.filterwarnings('ignore')

def load_data(PATH="UCI HAR Dataset/"):
    # Load HAR dataset and handle duplicate feature names
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
    
    # Load activity labels and create mapping
    activity_labels_df = pd.read_csv(PATH + "activity_labels.txt", sep="\\s+", header=None, names=["id", "activity"])
    activity_map = dict(zip(activity_labels_df.id, activity_labels_df.activity))
    
    # Load training and testing data
    X_train = pd.read_csv(PATH + "train/X_train.txt", sep="\\s+", header=None, names=feature_names)
    y_train = pd.read_csv(PATH + "train/y_train.txt", sep="\\s+", header=None, names=["Activity"])
    X_test = pd.read_csv(PATH + "test/X_test.txt", sep="\\s+", header=None, names=feature_names)
    y_test = pd.read_csv(PATH + "test/y_test.txt", sep="\\s+", header=None, names=["Activity"])
    
    # Map numeric activity IDs to descriptive labels
    y_train["Activity"] = y_train["Activity"].map(activity_map)
    y_test["Activity"] = y_test["Activity"].map(activity_map)
    
    return X_train, X_test, y_train, y_test

def to_binary_label(activity):
    # Convert activities to binary: 1 for movement activities, 0 for stationary
    return 1 if activity in ["WALKING", "WALKING_UPSTAIRS", "WALKING_DOWNSTAIRS"] else 0

def objective(trial, X_train, y_train):
    # Optuna objective function for hyperparameter optimization
    kernel = trial.suggest_categorical('kernel', ['linear', 'poly', 'rbf'])
    C = trial.suggest_float('C', 1e-3, 1e3, log=True)
    
    params = {'C': C, 'kernel': kernel, 'class_weight': 'balanced', 'random_state': 42}
    
    if kernel in ['poly', 'rbf']:
        params['gamma'] = trial.suggest_float('gamma', 1e-4, 1e1, log=True)
    if kernel == 'poly':
        params['degree'] = trial.suggest_int('degree', 2, 4)
    
    # Try with and without PCA
    use_pca = trial.suggest_categorical('use_pca', [True, False])
    pipeline_steps = [('scaler', StandardScaler())]
    
    if use_pca:
        variance_ratio = trial.suggest_float('variance_ratio', 0.75, 0.999) # A solid range for PCA
        pipeline_steps.append(('pca', PCA(n_components=variance_ratio)))
    
    pipeline_steps.append(('svc', SVC(**params)))
    pipeline = Pipeline(pipeline_steps)
    
    try:
        return cross_val_score(pipeline, X_train, y_train, cv=5, scoring='accuracy', n_jobs=1).mean()
    except Exception as e:
        print(f"Error in trial: {e}")
        return 0.0

def plot_confusion_matrix(cm, classes):
    # Visualize confusion matrix as heatmap
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=classes, yticklabels=classes)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.show()

def plot_roc_curve(y_true, y_prob):
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    roc_auc = auc(fpr, tpr)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:.2f})')
    plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic')
    plt.legend(loc="lower right")
    plt.show()

def plot_feature_importance(feature_names, importance):
    # Plot top 15 features
    top_n = 15
    idx = np.argsort(importance)[-top_n:]
    
    plt.figure(figsize=(10, 6))
    plt.barh(range(top_n), importance[idx])
    plt.yticks(range(top_n), [feature_names[i] for i in idx])
    plt.xlabel('Feature Importance')
    plt.title(f'Top {top_n} Most Important Features')
    plt.tight_layout()
    plt.show()

def plot_learning_curve(estimator, X, y, cv=5):
    train_sizes, train_scores, test_scores = learning_curve(
        estimator, X, y, cv=cv, n_jobs=1, 
        train_sizes=np.linspace(0.1, 1.0, 10),
        scoring='accuracy'
    )
    
    # Calculate mean and std for training and test scores
    train_mean = np.mean(train_scores, axis=1)
    train_std = np.std(train_scores, axis=1)
    test_mean = np.mean(test_scores, axis=1)
    test_std = np.std(test_scores, axis=1)
    
    # Plot learning curve
    plt.figure(figsize=(10, 6))
    plt.grid()
    plt.fill_between(train_sizes, train_mean - train_std,
                     train_mean + train_std, alpha=0.1, color="r")
    plt.fill_between(train_sizes, test_mean - test_std,
                     test_mean + test_std, alpha=0.1, color="g")
    plt.plot(train_sizes, train_mean, 'o-', color="r", label="Training score")
    plt.plot(train_sizes, test_mean, 'o-', color="g", label="Cross-validation score")
    plt.xlabel("Training examples")
    plt.ylabel("Accuracy")
    plt.title("Learning Curve")
    plt.legend(loc="best")
    plt.tight_layout()
    plt.show()
    
    print(f"\nFinal training score: {train_mean[-1]:.4f}")
    print(f"Final validation score: {test_mean[-1]:.4f}")
    print(f"Gap between training and validation: {train_mean[-1] - test_mean[-1]:.4f}")

def main():
    X_train, X_test, y_train, y_test = load_data()
    
    print(f"Data shapes - Train: {X_train.shape}, Test: {X_test.shape}")
    
    # Convert to binary classification problem (moving vs. stationary)
    y_train_binary = y_train["Activity"].apply(to_binary_label)
    y_test_binary = y_test["Activity"].apply(to_binary_label)
    
    print("\nClass distribution:")
    print(y_train_binary.value_counts(normalize=True))
    
    # Optimize hyperparameters with Optuna
    study = optuna.create_study(direction='maximize')
    study.optimize(lambda trial: objective(trial, X_train, y_train_binary), n_trials=30, n_jobs=1)
    
    best_params = study.best_params
    print(f"\nBest params: {best_params}")
    print(f"Best CV accuracy: {study.best_value:.4f}")
    
    # Build final model with best parameters
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
    
    # Train and evaluate model
    final_pipeline.fit(X_train, y_train_binary)
    y_pred = final_pipeline.predict(X_test)
    
    print("\nTest Results:")
    cm = confusion_matrix(y_test_binary, y_pred)
    print(cm)
    plot_confusion_matrix(cm, classes=['Stationary', 'Moving'])
    print("\nClassification Report:")
    print(classification_report(y_test_binary, y_pred))
    
    plot_learning_curve(final_pipeline, X_train, y_train_binary)
    
    # Plot ROC curve with different methods based on kernel type
    if hasattr(final_pipeline, 'decision_function'):
        y_scores = final_pipeline.decision_function(X_test)
        plot_roc_curve(y_test_binary, y_scores)
    elif hasattr(final_pipeline, 'predict_proba'):
        y_scores = final_pipeline.predict_proba(X_test)[:, 1]
        plot_roc_curve(y_test_binary, y_scores)
    
    # For linear kernels, extract feature importance
    if best_params['kernel'] == 'linear':
        svc = final_pipeline.named_steps['svc']
        feature_importance = np.abs(svc.coef_[0])
        
        # If PCA was used, transform feature importance
        if 'pca' in final_pipeline.named_steps:
            pca = final_pipeline.named_steps['pca']
            feature_importance = np.abs(svc.coef_[0] @ pca.components_)
        
        plot_feature_importance(X_train.columns, feature_importance)
        
        # Print top features
        top_features = np.argsort(feature_importance)[-10:][::-1]
        print("\nTop 10 Features:")
        for idx in top_features:
            print(f"{X_train.columns[idx]}: {feature_importance[idx]:.4f}")

if __name__ == "__main__":
    main() 