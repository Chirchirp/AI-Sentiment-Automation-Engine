#!/usr/bin/env python3
"""
Sentiment Automation CLI
Command-line interface for batch sentiment analysis

Usage:
    python sentiment_cli.py train input.csv --text-col text --label-col sentiment
    python sentiment_cli.py predict input.csv output.csv --text-col text
    python sentiment_cli.py evaluate input.csv --text-col text --label-col sentiment
"""

import argparse
import pandas as pd
import pickle
import json
import os
import sys
from datetime import datetime

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_recall_fscore_support, classification_report


# Configuration
MODEL_DIR = "models"
MODEL_FILE = os.path.join(MODEL_DIR, "sentiment_model.pkl")
VECTORIZER_FILE = os.path.join(MODEL_DIR, "vectorizer.pkl")
METADATA_FILE = os.path.join(MODEL_DIR, "model_metadata.json")


def ensure_model_dir():
    """Ensure models directory exists"""
    os.makedirs(MODEL_DIR, exist_ok=True)


def train_model(input_file, text_col, label_col):
    """Train sentiment model from labelled data"""
    print(f"Loading training data from {input_file}...")
    
    # Load data
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file)
    
    print(f"Loaded {len(df)} rows")
    
    # Prepare data
    texts = df[text_col].astype(str).tolist()
    labels = df[label_col].str.lower().tolist()
    
    # Validate labels
    valid_labels = {'positive', 'neutral', 'negative'}
    unique_labels = set(labels)
    
    if not unique_labels.issubset(valid_labels):
        print(f"Warning: Found invalid labels: {unique_labels - valid_labels}")
        print("Filtering to valid labels only...")
        valid_mask = df[label_col].str.lower().isin(valid_labels)
        df = df[valid_mask]
        texts = df[text_col].astype(str).tolist()
        labels = df[label_col].str.lower().tolist()
    
    print(f"Training with {len(texts)} samples...")
    
    # Split data
    X_train, X_test, y_train, y_test = train_test_split(
        texts, labels, test_size=0.2, random_state=42, stratify=labels
    )
    
    # Create vectorizer
    vectorizer = TfidfVectorizer(
        max_features=5000,
        ngram_range=(1, 2),
        min_df=2,
        max_df=0.95,
        stop_words='english'
    )
    
    # Transform
    X_train_tfidf = vectorizer.fit_transform(X_train)
    X_test_tfidf = vectorizer.transform(X_test)
    
    # Train model
    model = LogisticRegression(
        max_iter=1000,
        class_weight='balanced',
        random_state=42
    )
    model.fit(X_train_tfidf, y_train)
    
    # Evaluate
    y_pred = model.predict(X_test_tfidf)
    accuracy = accuracy_score(y_test, y_pred)
    precision, recall, f1, _ = precision_recall_fscore_support(
        y_test, y_pred, average='weighted', zero_division=0
    )
    
    # Save model
    ensure_model_dir()
    
    with open(MODEL_FILE, 'wb') as f:
        pickle.dump(model, f)
    
    with open(VECTORIZER_FILE, 'wb') as f:
        pickle.dump(vectorizer, f)
    
    metadata = {
        'accuracy': accuracy,
        'precision': precision,
        'recall': recall,
        'f1_score': f1,
        'train_size': len(X_train),
        'test_size': len(X_test),
        'trained_at': datetime.now().isoformat()
    }
    
    with open(METADATA_FILE, 'w') as f:
        json.dump(metadata, f, indent=2)
    
    print("\n" + "="*60)
    print("MODEL TRAINED SUCCESSFULLY")
    print("="*60)
    print(f"Accuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    print(f"\nTrain size: {len(X_train)}")
    print(f"Test size:  {len(X_test)}")
    print(f"\nModel saved to: {MODEL_FILE}")
    print("="*60)


def load_model():
    """Load trained model and vectorizer"""
    try:
        with open(MODEL_FILE, 'rb') as f:
            model = pickle.load(f)
        with open(VECTORIZER_FILE, 'rb') as f:
            vectorizer = pickle.load(f)
        with open(METADATA_FILE, 'r') as f:
            metadata = json.load(f)
        return model, vectorizer, metadata
    except FileNotFoundError:
        print("Error: No trained model found. Train a model first with 'train' command.")
        sys.exit(1)


def predict(input_file, output_file, text_col):
    """Predict sentiments for unlabelled data"""
    print(f"Loading data from {input_file}...")
    
    # Load data
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file)
    
    print(f"Loaded {len(df)} rows")
    
    # Load model
    print("Loading model...")
    model, vectorizer, metadata = load_model()
    print(f"Model accuracy: {metadata['accuracy']:.2%}")
    
    # Prepare texts
    texts = df[text_col].astype(str).tolist()
    
    # Predict
    print("Predicting sentiments...")
    X_tfidf = vectorizer.transform(texts)
    predictions = model.predict(X_tfidf)
    probabilities = model.predict_proba(X_tfidf)
    
    # Add to dataframe
    df['sentiment'] = predictions
    df['confidence'] = probabilities.max(axis=1)
    
    # Add class probabilities
    classes = model.classes_
    for i, cls in enumerate(classes):
        df[f'prob_{cls}'] = probabilities[:, i]
    
    # Save
    print(f"Saving results to {output_file}...")
    if output_file.endswith('.csv'):
        df.to_csv(output_file, index=False)
    else:
        df.to_excel(output_file, index=False)
    
    # Statistics
    sentiment_counts = df['sentiment'].value_counts()
    avg_confidence = df['confidence'].mean()
    
    print("\n" + "="*60)
    print("PREDICTION COMPLETE")
    print("="*60)
    print(f"Processed: {len(df)} rows")
    print(f"\nSentiment Distribution:")
    for sentiment, count in sentiment_counts.items():
        print(f"  {sentiment.title()}: {count} ({count/len(df)*100:.1f}%)")
    print(f"\nAverage Confidence: {avg_confidence:.2%}")
    print(f"\nResults saved to: {output_file}")
    print("="*60)


def evaluate(input_file, text_col, label_col):
    """Evaluate model on labelled test data"""
    print(f"Loading test data from {input_file}...")
    
    # Load data
    if input_file.endswith('.csv'):
        df = pd.read_csv(input_file)
    else:
        df = pd.read_excel(input_file)
    
    print(f"Loaded {len(df)} rows")
    
    # Load model
    print("Loading model...")
    model, vectorizer, metadata = load_model()
    
    # Prepare data
    texts = df[text_col].astype(str).tolist()
    true_labels = df[label_col].str.lower().tolist()
    
    # Predict
    print("Evaluating model...")
    X_tfidf = vectorizer.transform(texts)
    predictions = model.predict(X_tfidf)
    
    # Calculate metrics
    accuracy = accuracy_score(true_labels, predictions)
    precision, recall, f1, _ = precision_recall_fscore_support(
        true_labels, predictions, average='weighted', zero_division=0
    )
    
    report = classification_report(true_labels, predictions, zero_division=0)
    
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Test samples: {len(texts)}")
    print(f"\nAccuracy:  {accuracy:.2%}")
    print(f"Precision: {precision:.2%}")
    print(f"Recall:    {recall:.2%}")
    print(f"F1-Score:  {f1:.2%}")
    print("\n" + "-"*60)
    print("CLASSIFICATION REPORT")
    print("-"*60)
    print(report)
    print("="*60)


def main():
    parser = argparse.ArgumentParser(
        description='Sentiment Automation CLI',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train model
  python sentiment_cli.py train data.csv --text-col review --label-col sentiment
  
  # Predict sentiments
  python sentiment_cli.py predict input.csv output.csv --text-col review
  
  # Evaluate model
  python sentiment_cli.py evaluate test.csv --text-col review --label-col sentiment
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Command to run')
    
    # Train command
    train_parser = subparsers.add_parser('train', help='Train sentiment model')
    train_parser.add_argument('input', help='Input CSV/Excel file with labelled data')
    train_parser.add_argument('--text-col', required=True, help='Text column name')
    train_parser.add_argument('--label-col', required=True, help='Sentiment label column name')
    
    # Predict command
    predict_parser = subparsers.add_parser('predict', help='Predict sentiments')
    predict_parser.add_argument('input', help='Input CSV/Excel file')
    predict_parser.add_argument('output', help='Output CSV/Excel file')
    predict_parser.add_argument('--text-col', required=True, help='Text column name')
    
    # Evaluate command
    eval_parser = subparsers.add_parser('evaluate', help='Evaluate model on test data')
    eval_parser.add_argument('input', help='Input CSV/Excel file with labels')
    eval_parser.add_argument('--text-col', required=True, help='Text column name')
    eval_parser.add_argument('--label-col', required=True, help='True label column name')
    
    args = parser.parse_args()
    
    if args.command == 'train':
        train_model(args.input, args.text_col, args.label_col)
    elif args.command == 'predict':
        predict(args.input, args.output, args.text_col)
    elif args.command == 'evaluate':
        evaluate(args.input, args.text_col, args.label_col)
    else:
        parser.print_help()


if __name__ == '__main__':
    main()
