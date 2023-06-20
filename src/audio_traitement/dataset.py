from sklearn.model_selection import train_test_split

def create_dataset(features, labels, test_size=0.2, random_state=42):
    # Diviser les données en ensembles d'entraînement et de test
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=test_size, random_state=random_state)

    return X_train, X_test, y_train, y_test

def binary_encode(labels):
    encoded_labels = [1 if label == "oui" else 0 for label in labels]
    return encoded_labels