from sklearn.svm import SVC

def train_model(X_train, y_train):
    # Créer un modèle de classification (par exemple, SVM)
    model = SVC()

    # Entraîner le modèle sur l'ensemble d'entraînement
    model.fit(X_train, y_train)

    return model