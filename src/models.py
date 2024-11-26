from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.model_selection import GridSearchCV

def train_and_evaluate_rf(train_landscapes, train_labels, test_landscapes, test_labels, param_grid=None, cv_folds=3, random_state=42):
    """
    Train and evaluate a Random Forest classifier with Grid Search for hyperparameter tuning.

    Parameters:
    - train_landscapes: np.ndarray, training data.
    - train_labels: np.ndarray, labels for the training data.
    - test_landscapes: np.ndarray, testing data.
    - test_labels: np.ndarray, labels for the testing data.
    - param_grid: dict, parameter grid for Grid Search. Default is None.
    - cv_folds: int, number of cross-validation folds. Default is 3.
    - random_state: int, random state for reproducibility. Default is 42.

    Returns:
    - best_rf_model: Trained Random Forest model.
    - test_accuracy: Accuracy on the test set.
    - classification_rep: Classification report.
    - conf_matrix: Confusion matrix.
    """
    # Default parameter grid if not provided
    if param_grid is None:
        param_grid = {
            'n_estimators': [50, 100, 200],
            'max_depth': [None, 10, 20],
            'min_samples_split': [2, 5, 10],
            'min_samples_leaf': [1, 2, 4],
        }

    # Perform Grid Search with Cross-Validation
    grid_search = GridSearchCV(
        RandomForestClassifier(random_state=random_state),
        param_grid,
        cv=cv_folds,
        scoring='accuracy'
    )
    grid_search.fit(train_landscapes, train_labels)

    # Get the best model from Grid Search
    best_rf_model = grid_search.best_estimator_
    print("Best parameters:", grid_search.best_params_)

    # Train the best model on training data
    best_rf_model.fit(train_landscapes, train_labels)

    # Predictions on the test set
    test_predictions = best_rf_model.predict(test_landscapes)

    # Evaluate the model
    test_accuracy = accuracy_score(test_labels, test_predictions)
    classification_rep = classification_report(test_labels, test_predictions)
    conf_matrix = confusion_matrix(test_labels, test_predictions)

    # Display results
    print(f"Test Accuracy: {test_accuracy:.4f}")
    print("\nClassification Report:\n", classification_rep)
    print("\nConfusion Matrix:\n", conf_matrix)

    return best_rf_model, test_accuracy, classification_rep, conf_matrix
