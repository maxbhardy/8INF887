import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
import numpy as np


class LogisticRegression(nn.Module):
    def __init__(self, input_dim):
        super(LogisticRegression, self).__init__()
        self.linear = nn.Linear(input_dim, 1)

    def forward(self, x):
        return torch.sigmoid(self.linear(x))


# Charger et préparer les données Iris
iris = load_iris()
X = iris.data
y = iris.target

# Filtrer pour une classification binaire (classes 0 et 1 uniquement)
binary_filter = y < 2  # On garde les classes 0 et 1
X = X[binary_filter]
y = y[binary_filter]

# Convertir les étiquettes en un format compatible avec BCE (0 ou 1)
y = y.astype(np.float32).reshape(-1, 1)

# Diviser en ensembles d'entraînement et de test
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

# Normaliser les données (centrer et réduire)
scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

# Convertir en tenseurs PyTorch
X_train = torch.tensor(X_train, dtype=torch.float32)
y_train = torch.tensor(y_train, dtype=torch.float32)
X_test = torch.tensor(X_test, dtype=torch.float32)
y_test = torch.tensor(y_test, dtype=torch.float32)

# Définir le modèle de régression logistique
model = LogisticRegression(input_dim=X_train.shape[1])

# Définir la fonction de perte et l'optimiseur
criterion = nn.BCELoss()  # Binary Cross-Entropy Loss
optimizer = optim.SGD(model.parameters(), lr=0.01)

# Boucle d'entraînement
num_epochs = 100
for epoch in range(num_epochs):
    # Forward pass
    outputs = model(X_train)
    loss = criterion(outputs, y_train)

    # Backward pass
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    # Affichage périodique de la perte
    if (epoch + 1) % 10 == 0:
        print(f"Epoch [{epoch + 1}/{num_epochs}], Loss: {loss.item():.4f}")

# Tester le modèle
with torch.no_grad():
    test_outputs = model(X_test)
    test_predictions = (test_outputs > 0.5).float()  # Seuil à 0.5
    accuracy = (test_predictions == y_test).float().mean()
    print("\nPrécision sur l'ensemble de test :", accuracy.item())

# Afficher les prédictions et les probabilités
print("\nPrédictions :")
print(test_predictions.numpy())
print("Probabilités :")
print(test_outputs.numpy())
