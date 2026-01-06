"""Train invoice classification model with comprehensive evaluation."""

import pandas as pd
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import OneHotEncoder, LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
    confusion_matrix,
)
import lightgbm as lgb

from config import (
    MODEL_PATH,
    DATA_DIR,
    FEATURES,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DATETIME_FEATURES,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    LGBM_PARAMS,
)


def load_and_prepare_data(csv_path: str) -> tuple:
    """Load and prepare training data from CSV.

    Args:
        csv_path: Path to CSV file exported from database

    Returns:
        Tuple of (X, y) features and target
    """
    print(f"Loading data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} records")
    print(f"Classes distribution:\n{df['expenseCategory'].value_counts()}")

    # Drop rows with missing title_normalized (critical feature)
    df.dropna(subset=['title_normalized'], inplace=True)
    print(f"After dropping missing titles: {len(df)} records")

    # Process issueDate
    df['issueDate'] = pd.to_datetime(df['issueDate'])
    df['issueMonth'] = df['issueDate'].dt.month
    df['issueYear'] = df['issueDate'].dt.year
    df['issueDay'] = df['issueDate'].dt.day

    # Calculate VAT features
    df['VAT_Amount'] = df['grossPrice'] - df['netPrice']
    df['VAT_Rate'] = (df['grossPrice'] / df['netPrice']) - 1

    # Handle infinite values from VAT_Rate calculation
    df['VAT_Rate'].replace([float('inf'), float('-inf')], 0, inplace=True)

    # Split features and target
    X = df[FEATURES]
    y = df['expenseCategory']

    return X, y, df


def create_preprocessing_pipeline():
    """Create sklearn preprocessing pipeline.

    Returns:
        ColumnTransformer for preprocessing
    """
    # Numerical transformer: impute with median
    numerical_transformer = SimpleImputer(strategy='median')

    # Categorical transformer: impute + one-hot encode
    # Using sparse=False for compatibility, handle_unknown='ignore' for robustness
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=100))
    ])

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, NUMERICAL_FEATURES),
            ('cat', categorical_transformer, CATEGORICAL_FEATURES),
            ('datetime', 'passthrough', DATETIME_FEATURES)
        ]
    )

    return preprocessor


def train_and_evaluate(X, y):
    """Train model with cross-validation and comprehensive evaluation.

    Args:
        X: Features DataFrame
        y: Target Series

    Returns:
        Trained pipeline and evaluation metrics
    """
    print("\n" + "="*60)
    print("TRAINING INVOICE CLASSIFIER")
    print("="*60)

    # Encode labels for LightGBM (requires 0-indexed integers)
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline()

    # Create LightGBM classifier
    # Note: LightGBM is faster and lighter than sklearn's GradientBoosting
    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Add label encoder to pipeline for saving
    pipeline.label_encoder = label_encoder

    # Cross-validation
    print(f"\nPerforming {CV_FOLDS}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy')

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train-test split for detailed evaluation
    print("\nSplitting data for train/test evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train on training set
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    # Evaluate on test set
    print("\nEvaluating on test set...")
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    # Get class names back for reporting
    class_names = label_encoder.classes_

    print("\n" + "="*60)
    print("TEST SET PERFORMANCE")
    print("="*60)
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Weighted Precision: {precision:.4f}")
    print(f"Weighted Recall: {recall:.4f}")
    print(f"Weighted F1-Score: {f1:.4f}")

    print("\nDetailed Classification Report:")
    print(classification_report(y_test, y_pred, target_names=class_names, zero_division=0))

    # Feature importance (if available)
    if hasattr(model, 'feature_importances_'):
        print("\nTop 10 Most Important Features:")
        feature_names = pipeline.named_steps['preprocessor'].get_feature_names_out()
        importances = model.feature_importances_
        indices = importances.argsort()[-10:][::-1]

        for i, idx in enumerate(indices, 1):
            print(f"{i}. {feature_names[idx]}: {importances[idx]:.4f}")

    # Compile metrics
    metrics = {
        'training_date': datetime.now().isoformat(),
        'n_samples': len(X),
        'n_train': len(X_train),
        'n_test': len(X_test),
        'n_classes': len(class_names),
        'classes': class_names.tolist(),
        'cv_mean_accuracy': float(cv_scores.mean()),
        'cv_std_accuracy': float(cv_scores.std()),
        'cv_scores': cv_scores.tolist(),
        'test_accuracy': float(accuracy),
        'test_precision': float(precision),
        'test_recall': float(recall),
        'test_f1': float(f1),
    }

    # Train on full dataset for deployment
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print("="*60)
    pipeline.fit(X, y_encoded)

    return pipeline, metrics


def save_model_and_metrics(pipeline, metrics):
    """Save trained model and evaluation metrics.

    Args:
        pipeline: Trained sklearn pipeline
        metrics: Dictionary of evaluation metrics
    """
    print(f"\nSaving model to {MODEL_PATH}...")
    joblib.dump(pipeline, MODEL_PATH)

    metrics_path = MODEL_PATH.parent / "model_metrics.json"
    print(f"Saving metrics to {metrics_path}...")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*60)
    print("MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved: {MODEL_PATH}")
    print(f"Model size: {MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Metrics saved: {metrics_path}")


def main(csv_file: str = "invoices_training_data.csv"):
    """Main training pipeline.

    Args:
        csv_file: Name of CSV file in data directory
    """
    csv_path = DATA_DIR / csv_file

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {csv_path}\n"
            f"Please export your SQL query results to {csv_path}"
        )

    # Load and prepare data
    X, y, df = load_and_prepare_data(csv_path)

    # Train and evaluate
    pipeline, metrics = train_and_evaluate(X, y)

    # Save model and metrics
    save_model_and_metrics(pipeline, metrics)

    print("\n✓ Training complete! You can now deploy the model.")
    print(f"  Test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"  CV accuracy: {metrics['cv_mean_accuracy']:.2%} ± {metrics['cv_std_accuracy']:.2%}")


if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "invoices_training_data.csv"
    main(csv_file)
