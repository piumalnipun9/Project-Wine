import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import joblib
import os
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix, classification_report, roc_curve, auc

def main():
    # Setup directories
    os.makedirs('../report_task1/figures', exist_ok=True)
    os.makedirs('../models', exist_ok=True)
    
    # ---------------------------------------------------------
    # PHASE 2: Data Pipeline and Feature Handling
    # ---------------------------------------------------------
    print("Loading data...")
    df = pd.read_csv('../data/WineQT.csv')
    
    print(f"Dataset shape: {df.shape}")
    print("Checking for missing values:")
    print(df.isnull().sum())
    
    # Drop the 'Id' column as it's not a useful feature
    if 'Id' in df.columns:
        df = df.drop('Id', axis=1)
        
    # Correlation Heatmap
    plt.figure(figsize=(12, 10))
    sns.heatmap(df.corr(), annot=True, cmap='coolwarm', fmt=".2f")
    plt.title("Feature Correlation Heatmap")
    plt.tight_layout()
    plt.savefig('../report_task1/figures/correlation_heatmap.png')
    plt.close()
    
    # Target distribution before binarization
    plt.figure(figsize=(8, 5))
    sns.countplot(x='quality', data=df, palette='viridis')
    plt.title("Distribution of Wine Quality (Original)")
    plt.savefig('../report_task1/figures/quality_distribution.png')
    plt.close()
    
    # Framing: Convert to Binary Classification (Good vs Bad)
    # Good quality >= 6, Bad < 6
    df['is_good'] = (df['quality'] >= 6).astype(int)
    
    # Distribution after binarization
    plt.figure(figsize=(6, 4))
    sns.countplot(x='is_good', data=df, palette='Set2')
    plt.title("Distribution of Wine Quality (Binary: >= 6 is Good)")
    plt.xticks([0, 1], ['Bad (0)', 'Good (1)'])
    plt.savefig('../report_task1/figures/binary_distribution.png')
    plt.close()
    
    # Feature scaling and splitting
    X = df.drop(['quality', 'is_good'], axis=1)
    y = df['is_good']
    
    # Train-test split
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42, stratify=y)
    
    # Scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # Save the scaler for Streamlit deployment
    joblib.dump(scaler, '../models/scaler.joblib')
    
    # ---------------------------------------------------------
    # PHASE 3: Modeling, Debugging, and Evaluation
    # ---------------------------------------------------------
    
    # DEBUGGING SCENARIO (Intentional Bug for Report):
    # Imagine we forgot to scale the data for Logistic Regression, which is sensitive to scaling.
    # bug_lr = LogisticRegression(max_iter=100)
    # bug_lr.fit(X_train, y_train) # Training on unscaled data
    # print("Accuracy with bug (unscaled data):", bug_lr.score(X_test, y_test))
    # We will document this in the report and fix it by using X_train_scaled.
    
    print("\nTraining Baseline Models...")
    
    # Model 1: Logistic Regression
    lr_model = LogisticRegression(random_state=42)
    lr_model.fit(X_train_scaled, y_train)
    lr_preds = lr_model.predict(X_test_scaled)
    lr_probs = lr_model.predict_proba(X_test_scaled)[:, 1]
    
    # Model 2: Random Forest (with GridSearchCV)
    rf = RandomForestClassifier(random_state=42)
    rf_params = {
        'n_estimators': [50, 100],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5]
    }
    grid_search = GridSearchCV(rf, rf_params, cv=5, scoring='accuracy', n_jobs=-1)
    grid_search.fit(X_train_scaled, y_train)
    
    best_rf = grid_search.best_estimator_
    print(f"Best RF Parameters: {grid_search.best_params_}")
    
    rf_preds = best_rf.predict(X_test_scaled)
    rf_probs = best_rf.predict_proba(X_test_scaled)[:, 1]
    
    # Save the best model
    joblib.dump(best_rf, '../models/random_forest_model.joblib')
    
    # Evaluation Metrics
    def print_metrics(name, y_true, y_pred):
        print(f"\n--- {name} ---")
        print(f"Accuracy:  {accuracy_score(y_true, y_pred):.4f}")
        print(f"Precision: {precision_score(y_true, y_pred):.4f}")
        print(f"Recall:    {recall_score(y_true, y_pred):.4f}")
        print(f"F1-score:  {f1_score(y_true, y_pred):.4f}")
        
    print_metrics("Logistic Regression", y_test, lr_preds)
    print_metrics("Random Forest", y_test, rf_preds)
    
    # Confusion Matrices
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    sns.heatmap(confusion_matrix(y_test, lr_preds), annot=True, fmt='d', cmap='Blues', ax=axes[0])
    axes[0].set_title('Logistic Regression Confusion Matrix')
    axes[0].set_xlabel('Predicted')
    axes[0].set_ylabel('Actual')
    
    sns.heatmap(confusion_matrix(y_test, rf_preds), annot=True, fmt='d', cmap='Greens', ax=axes[1])
    axes[1].set_title('Random Forest Confusion Matrix')
    axes[1].set_xlabel('Predicted')
    axes[1].set_ylabel('Actual')
    
    plt.tight_layout()
    plt.savefig('../report_task1/figures/confusion_matrices.png')
    plt.close()
    
    # ROC Curves
    fpr_lr, tpr_lr, _ = roc_curve(y_test, lr_probs)
    roc_auc_lr = auc(fpr_lr, tpr_lr)
    
    fpr_rf, tpr_rf, _ = roc_curve(y_test, rf_probs)
    roc_auc_rf = auc(fpr_rf, tpr_rf)
    
    plt.figure(figsize=(8, 6))
    plt.plot(fpr_lr, tpr_lr, color='blue', lw=2, label=f'Logistic Regression (AUC = {roc_auc_lr:.2f})')
    plt.plot(fpr_rf, tpr_rf, color='green', lw=2, label=f'Random Forest (AUC = {roc_auc_rf:.2f})')
    plt.plot([0, 1], [0, 1], color='gray', lw=2, linestyle='--')
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('Receiver Operating Characteristic (ROC) Curve')
    plt.legend(loc="lower right")
    plt.savefig('../report_task1/figures/roc_curves.png')
    plt.close()
    
    # Feature Importance (Random Forest)
    importances = best_rf.feature_importances_
    indices = np.argsort(importances)[::-1]
    
    plt.figure(figsize=(10, 6))
    plt.title("Feature Importances (Random Forest)")
    plt.bar(range(X.shape[1]), importances[indices], align="center")
    plt.xticks(range(X.shape[1]), X.columns[indices], rotation=45, ha='right')
    plt.xlim([-1, X.shape[1]])
    plt.tight_layout()
    plt.savefig('../report_task1/figures/feature_importance.png')
    plt.close()
    
    print("\nAnalysis complete. Figures and models saved.")

if __name__ == '__main__':
    # When running this script directly from the src directory
    main()
