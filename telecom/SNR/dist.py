import matplotlib.pyplot as plt
import numpy as np

# Données
data = [
    [2, 29.23],
    [1, 35.11],
    [0.5, 41.38],
    [3, 25.58]
]

# Extraction des colonnes dans des vecteurs distincts
x = np.array([row[0] for row in data])
y_dB = np.array([row[1] for row in data])

# Conversion des valeurs dB en échelle linéaire
#y_linear = 10 ** (y_dB / 10)  # Conversion dB -> linéaire

# Ajustement polynomial de degré 2
coefficients = np.polyfit(x, y_dB, 2)
poly = np.poly1d(coefficients)

# Génération de points pour l'approximation
x_approx = np.linspace(min(x), max(x), 100)
y_approx = poly(x_approx)

# Création du tracé
plt.figure(figsize=(8, 6))

# Plot des données
plt.scatter(x, y_dB, label='mesures')

# Plot de l'approximation polynomiale
plt.plot(x_approx, y_approx, label='Approximation degré 2', color='red')

plt.xlabel('Distance [m]')
plt.ylabel('SNR [dB]')
plt.title('Comportement du SNR avec la distance')

plt.legend()
plt.grid(True)
plt.show()