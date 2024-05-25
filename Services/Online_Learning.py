import joblib
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV, StratifiedShuffleSplit

def train_incremental_random_forest(data, save_path):
    """
    Train an Incremental Random Forest model using the provided data and save it.

    Args:
    - data: DataFrame containing the feature-engineered data.
    - save_path: Path to save the trained model.

    Returns:
    - model: Trained Incremental Random Forest model.
    """
    # Assuming 'attack' column is the target variable
    X_train = data.drop(columns=["attack"])
    y_train = data["attack"]

    # Define the parameter grid
    param_grid = {
        'n_estimators': [100, 200, 300],
        'max_depth': [None, 10, 20],
        'min_samples_split': [2, 5, 10],
        'min_samples_leaf': [1, 2, 4]
    }

    # Initialize the Incremental Random Forest classifier
    model = RandomForestClassifier(random_state=42)

    # Initialize StratifiedShuffleSplit cross-validation
    cv = StratifiedShuffleSplit(n_splits=3, test_size=0.2, random_state=42)

    # Initialize Grid Search with cross-validation
    grid_search = GridSearchCV(estimator=model, param_grid=param_grid, cv=cv, n_jobs=-1, verbose=2)

    # Train the Grid Search model
    grid_search.fit(X_train, y_train)

    # Get the best parameters
    best_params = grid_search.best_params_
    print("Best Parameters:", best_params)

    # Use the best parameters to train the final model
    best_model = RandomForestClassifier(**best_params, random_state=42)
    best_model.fit(X_train, y_train)

    # Save the trained model
    joblib.dump(best_model, save_path)

    # Return the trained model
    return best_model