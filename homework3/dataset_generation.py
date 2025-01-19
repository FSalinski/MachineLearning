import numpy as np
import pandas as pd
from sklearn.datasets import make_circles, make_moons, make_blobs
import matplotlib.pyplot as plt
import seaborn as sns

sns.set(style='whitegrid')

def generate_data(type, n_samples, n_features, n_classes, noise, random_state):
    X, y = None, None

    if type == 'blobs':
        X, y = make_blobs(n_samples=n_samples, n_features=n_features, centers=n_classes, cluster_std=noise, random_state=random_state)
    elif type == 'moons':
        X, y = make_moons(n_samples=n_samples, noise=noise, random_state=random_state)
    elif type == 'circles':
        X, y = make_circles(n_samples=n_samples, noise=noise, random_state=random_state)

    return X, y

def draw_data(X, y):
    plt.figure(figsize=(10, 6))
    plt.scatter(X[:, 0], X[:, 1], c=y, s=50, cmap='viridis')
    plt.title('Generated Data')
    plt.show()

def save_data(X, y, filename):
    data = np.column_stack((y, X))
    df = pd.DataFrame(data, columns=['y', 'X1', 'X2'])
    df.to_csv(filename, index=False)

def main():
    n_samples = 500
    n_features = 2
    n_classes = 7
    noise = 0.8
    random_state = 123

    X, y = generate_data('blobs', n_samples, n_features, n_classes, noise, random_state)
    draw_data(X, y)
    save_data(X, y, '327217_data.csv')

if __name__ == '__main__':
    main()
