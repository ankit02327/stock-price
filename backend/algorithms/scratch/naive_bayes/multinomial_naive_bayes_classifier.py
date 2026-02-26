"""
Multinomial Naive Bayes Classifier Implementation from Scratch

This module implements a Multinomial Naive Bayes classifier using:
- Multinomial probability distribution
- Laplace smoothing (add-one smoothing)
- Text classification
- Binary classification

Mathematical Foundation:
- Multinomial: P(x|y) = (count(x,y) + α) / (count(y) + α * |V|)
- Laplace Smoothing: α = 1 (add-one smoothing)
- Bayes' Theorem: P(y|x) = P(x|y) * P(y) / P(x)
- Naive Assumption: P(x|y) = ∏ P(xᵢ|y)
"""

import math
from collections import defaultdict, Counter

def calculate_metrics(y_true, y_pred):
    """
    Calculate classification metrics
    
    Parameters:
    - y_true: true class labels
    - y_pred: predicted class labels
    
    Returns:
    - accuracy, precision, recall, f1: classification metrics
    """
    # Calculate confusion matrix components
    tp = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 1)  # True Positives
    fp = sum(1 for true, pred in zip(y_true, y_pred) if true == 0 and pred == 1)  # False Positives
    fn = sum(1 for true, pred in zip(y_true, y_pred) if true == 1 and pred == 0)  # False Negatives
    
    # Calculate metrics
    accuracy = sum(1 for true, pred in zip(y_true, y_pred) if true == pred) / len(y_true)
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0
    
    return accuracy, precision, recall, f1

def multinomial_naive_bayes(X_train, y_train, X_test):
    """
    Multinomial Naive Bayes classifier implementation
    
    Parameters:
    - X_train: training documents (list of strings)
    - y_train: training labels
    - X_test: test documents (list of strings)
    
    Returns:
    - predictions: predicted class labels
    - class_priors: prior probabilities for each class
    """
    # Calculate class priors
    class_counts = Counter(y_train)
    total_docs = len(y_train)
    class_priors = {cls: count/total_docs for cls, count in class_counts.items()}
    
    # Build vocabulary and count words per class
    vocabulary = set()
    word_counts = defaultdict(lambda: defaultdict(int))
    total_words_per_class = defaultdict(int)
    
    for doc, label in zip(X_train, y_train):
        words = doc.lower().split()
        for word in words:
            vocabulary.add(word)
            word_counts[label][word] += 1
            total_words_per_class[label] += 1
    
    vocab_size = len(vocabulary)
    
    # Calculate word probabilities with Laplace smoothing
    word_probs = defaultdict(lambda: defaultdict(float))
    for cls in class_counts.keys():
        for word in vocabulary:
            # Laplace smoothing: (count + 1) / (total + |V|)
            word_probs[cls][word] = (word_counts[cls][word] + 1) / (total_words_per_class[cls] + vocab_size)
    
    # Make predictions on test set
    predictions = []
    for doc in X_test:
        words = doc.lower().split()
        class_scores = {}
        
        for cls in class_counts.keys():
            # Start with log prior probability
            log_prob = math.log(class_priors[cls])
            # Add log likelihood for each word
            for word in words:
                if word in vocabulary:
                    log_prob += math.log(word_probs[cls][word])
            class_scores[cls] = log_prob
        
        # Choose class with highest log probability
        predicted_class = max(class_scores, key=class_scores.get)
        predictions.append(predicted_class)
    
    return predictions, class_priors

if __name__ == "__main__":
    # Read input data
    n = int(input())
    texts = []
    labels = []
    
    for _ in range(n):
        line = input().strip()
        parts = line.rsplit(' , ', 1)  # Split on last occurrence of ' , '
        text = parts[0]
        label = int(parts[1])
        texts.append(text)
        labels.append(label)
    
    # Split data into training and testing sets
    train_size = int(0.7 * n)
    X_train = texts[:train_size]
    y_train = labels[:train_size]
    X_test = texts[train_size:]
    y_test = labels[train_size:]
    
    # Train and test the classifier
    predictions, class_priors = multinomial_naive_bayes(X_train, y_train, X_test)
    
    # Calculate and display metrics
    accuracy, precision, recall, f1 = calculate_metrics(y_test, predictions)
    
    # Display results
    print(f"Training samples: {len(X_train)}")
    print(f"Test samples: {len(X_test)}")
    print(f"Class 0 Prior: {class_priors[0]:.2f}")
    print(f"Class 1 Prior: {class_priors[1]:.2f}")
    print(f"Predictions: {predictions}")
    print(f"Actual: {y_test}")
    print(f"Accuracy={accuracy:.2f}")
    print(f"Precision={precision:.2f}")
    print(f"Recall={recall:.2f}")
    print(f"F1={f1:.2f}")