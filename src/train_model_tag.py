"""Train invoice tag classification model with comprehensive evaluation.

This script trains a model to predict expense tags (e.g., 'benefit-training',
'legal-advice', 'accounting', etc.) based on invoice data.
"""

import pandas as pd
import joblib
import json
import warnings
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.pipeline import Pipeline
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
import lightgbm as lgb

from src.config import (
    TAG_MODEL_PATH,
    DATA_DIR,
    RANDOM_STATE,
    TEST_SIZE,
    CV_FOLDS,
    LGBM_PARAMS,
)
from src.preprocessing import (
    prepare_invoice_features,
    create_preprocessing_pipeline,
    NUMERICAL_FEATURES,
    CATEGORICAL_FEATURES,
    DATETIME_FEATURES,
    TEXT_FEATURE,
    ALL_FEATURES,
)


# Suppress benign sklearn warning about feature names in LightGBM pipeline
warnings.filterwarnings('ignore', message='X does not have valid feature names')


def load_and_prepare_data(csv_path: str) -> tuple:
    """Load and prepare tag training data from CSV.

    Args:
        csv_path: Path to CSV file exported from database

    Returns:
        Tuple of (X, y) features and target
    """
    print(f"Loading tag training data from {csv_path}...")
    df = pd.read_csv(csv_path)

    print(f"Loaded {len(df)} records")
    print(f"Tag distribution:\n{df['tag'].value_counts()}")

    # Prepare features (adds VAT and date features)
    df = prepare_invoice_features(df)
    print(f"After preprocessing: {len(df)} records")

    # Extract features and target
    X = df[ALL_FEATURES]
    y = df['tag']

    return X, y, df


def train_and_evaluate(X, y):
    """Train tag model with comprehensive evaluation."""

    print("\n" + "="*60)
    print("TRAINING INVOICE TAG CLASSIFIER")
    print("="*60)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(
        numerical_features=NUMERICAL_FEATURES,
        categorical_features=CATEGORICAL_FEATURES,
        datetime_features=DATETIME_FEATURES,
        text_feature=TEXT_FEATURE,
        max_tfidf_features=200  # Keep model light
    )

    # Create model
    model = lgb.LGBMClassifier(**LGBM_PARAMS)

    # Create full pipeline
    pipeline = Pipeline(steps=[
        ('preprocessor', preprocessor),
        ('model', model)
    ])

    # Store label encoder
    pipeline.label_encoder = label_encoder

    # Cross-validation
    print(f"\nPerforming {CV_FOLDS}-fold cross-validation...")
    cv = StratifiedKFold(n_splits=CV_FOLDS, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = cross_val_score(pipeline, X, y_encoded, cv=cv, scoring='accuracy')

    print(f"Cross-validation scores: {cv_scores}")
    print(f"Mean CV Accuracy: {cv_scores.mean():.4f} (+/- {cv_scores.std() * 2:.4f})")

    # Train-test split
    print("\nSplitting data for train/test evaluation...")
    X_train, X_test, y_train, y_test = train_test_split(
        X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
    )

    print(f"Training set size: {len(X_train)}")
    print(f"Test set size: {len(X_test)}")

    # Train
    print("\nTraining model...")
    pipeline.fit(X_train, y_train)

    # Evaluate
    print("\nEvaluating on test set...")
    y_pred = pipeline.predict(X_test)

    # Calculate metrics
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, support = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )

    # Get class names
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

    # Show TF-IDF features learned
    print("\nTop 20 TF-IDF terms learned:")
    vectorizer = pipeline.named_steps['preprocessor'].named_transformers_['text']
    feature_names = vectorizer.get_feature_names_out()
    print(f"Total TF-IDF features: {len(feature_names)}")
    print("Sample terms:", ', '.join(feature_names[:20]))

    # Metrics
    metrics = {
        'model_type': 'tag',
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
        'features': {
            'text_features': len(feature_names),
            'total_features': 'TF-IDF + numerical + categorical',
        }
    }

    # Train on full dataset
    print("\n" + "="*60)
    print("TRAINING FINAL MODEL ON FULL DATASET")
    print("="*60)
    pipeline.fit(X, y_encoded)

    return pipeline, metrics


def save_model_and_metrics(pipeline, metrics):
    """Save trained tag model and metrics."""

    print(f"\nSaving model to {TAG_MODEL_PATH}...")
    joblib.dump(pipeline, TAG_MODEL_PATH)

    metrics_path = TAG_MODEL_PATH.parent / "tag_model_metrics.json"
    print(f"Saving metrics to {metrics_path}...")
    with open(metrics_path, 'w') as f:
        json.dump(metrics, f, indent=2)

    print("\n" + "="*60)
    print("TAG MODEL TRAINING COMPLETE")
    print("="*60)
    print(f"Model saved: {TAG_MODEL_PATH}")
    print(f"Model size: {TAG_MODEL_PATH.stat().st_size / 1024 / 1024:.2f} MB")
    print(f"Metrics saved: {metrics_path}")


def main(csv_file: str = "invoices_tag_training_data.csv"):
    """Main training pipeline for tag model."""
    csv_path = DATA_DIR / csv_file

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {csv_path}\n"
            f"Please export tag training data using queries/fetch_tag_training_data.sql"
        )

    # Load data
    X, y, df = load_and_prepare_data(csv_path)

    # Train and evaluate
    pipeline, metrics = train_and_evaluate(X, y)

    # Save
    save_model_and_metrics(pipeline, metrics)

    print(f"\n✓ Tag model training complete!")
    print(f"  Test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"  CV accuracy: {metrics['cv_mean_accuracy']:.2%} ± {metrics['cv_std_accuracy']:.2%}")


if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "invoices_tag_training_data.csv"
    main(csv_file)
