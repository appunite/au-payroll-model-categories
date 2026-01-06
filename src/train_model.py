"""Train invoice classification model with comprehensive evaluation.

Uses TF-IDF vectorization of full invoice titles for rich text features,
combined with numerical, categorical, and temporal features.
"""

import pandas as pd
import joblib
import json
from datetime import datetime
from pathlib import Path
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import LabelEncoder
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics import (
    accuracy_score,
    precision_recall_fscore_support,
    classification_report,
)
import lightgbm as lgb

from config import (
    MODEL_PATH,
    DATA_DIR,
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

    # Drop rows with missing invoice_title (critical feature)
    df.dropna(subset=['invoice_title'], inplace=True)
    print(f"After dropping missing titles: {len(df)} records")

    # Process issueDate
    df['issueDate'] = pd.to_datetime(df['issueDate'])
    df['issueMonth'] = df['issueDate'].dt.month
    df['issueYear'] = df['issueDate'].dt.year
    df['issueDay'] = df['issueDate'].dt.day

    # Calculate VAT features
    df['VAT_Amount'] = df['grossPrice'] - df['netPrice']
    df['VAT_Rate'] = (df['grossPrice'] / df['netPrice']) - 1

    # Handle infinite values
    df.loc[df['VAT_Rate'].isin([float('inf'), float('-inf')]), 'VAT_Rate'] = 0

    # Features for this model
    numerical_features = ['netPrice', 'VAT_Amount', 'VAT_Rate']
    categorical_features = ['entityId', 'ownerId', 'currency', 'tin']
    datetime_features = ['issueYear', 'issueMonth', 'issueDay']
    text_feature = 'invoice_title'

    # Combine features
    X = df[numerical_features + categorical_features + datetime_features + [text_feature]]
    y = df['expenseCategory']

    return X, y, df, numerical_features, categorical_features, datetime_features, text_feature


def create_preprocessing_pipeline(numerical_features, categorical_features, datetime_features, text_feature):
    """Create sklearn preprocessing pipeline with TF-IDF for text.

    Returns:
        ColumnTransformer for preprocessing
    """
    # Numerical transformer
    numerical_transformer = SimpleImputer(strategy='median')

    # Categorical transformer (excluding title)
    from sklearn.preprocessing import OneHotEncoder
    categorical_transformer = Pipeline(steps=[
        ('imputer', SimpleImputer(strategy='constant', fill_value='MISSING')),
        ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False, max_categories=50))
    ])

    # Text transformer: TF-IDF on invoice titles
    # Key parameters:
    # - max_features: limit vocab size for faster inference
    # - ngram_range: capture 1-2 word phrases (e.g., "software license")
    # - min_df: ignore very rare terms (appear in <3 documents)
    # - max_df: ignore very common terms (appear in >80% of documents)
    text_transformer = TfidfVectorizer(
        max_features=200,  # Keep model light for fast cold starts
        ngram_range=(1, 2),  # Unigrams and bigrams
        min_df=3,  # Must appear in at least 3 documents
        max_df=0.8,  # Must not appear in >80% of documents
        lowercase=True,
        strip_accents='unicode',  # Handle Polish characters
        token_pattern=r'\b[a-zA-Z]{2,}\b',  # Words with 2+ letters
    )

    # Combine transformers
    preprocessor = ColumnTransformer(
        transformers=[
            ('num', numerical_transformer, numerical_features),
            ('cat', categorical_transformer, categorical_features),
            ('datetime', 'passthrough', datetime_features),
            ('text', text_transformer, text_feature),
        ]
    )

    return preprocessor


def train_and_evaluate(X, y, numerical_features, categorical_features, datetime_features, text_feature):
    """Train model with TF-IDF and comprehensive evaluation."""

    print("\n" + "="*60)
    print("TRAINING INVOICE CLASSIFIER")
    print("="*60)

    # Encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y)

    # Create preprocessing pipeline
    preprocessor = create_preprocessing_pipeline(
        numerical_features, categorical_features, datetime_features, text_feature
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
    """Save trained model and metrics."""

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
    """Main training pipeline."""
    csv_path = DATA_DIR / csv_file

    if not csv_path.exists():
        raise FileNotFoundError(
            f"Training data not found at {csv_path}\n"
            f"Please run: make fetch-data"
        )

    # Load data
    X, y, df, numerical_features, categorical_features, datetime_features, text_feature = load_and_prepare_data(csv_path)

    # Train and evaluate
    pipeline, metrics = train_and_evaluate(X, y, numerical_features, categorical_features, datetime_features, text_feature)

    # Save
    save_model_and_metrics(pipeline, metrics)

    print("\n✓ Training complete! You can now deploy the model.")
    print(f"  Test accuracy: {metrics['test_accuracy']:.2%}")
    print(f"  CV accuracy: {metrics['cv_mean_accuracy']:.2%} ± {metrics['cv_std_accuracy']:.2%}")


if __name__ == "__main__":
    import sys
    csv_file = sys.argv[1] if len(sys.argv) > 1 else "invoices_training_data.csv"
    main(csv_file)
