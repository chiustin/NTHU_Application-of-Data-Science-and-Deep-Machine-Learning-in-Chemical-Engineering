import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import ListedColormap

def plot_decision_regions(X, y, classifier, test_idx=None, resolution=0.02):
#     setup marker generator and color map
    markers = ('s', 'x', 'o', '^', 'v')
    colors = ('red', 'blue', 'lightgreen', 'gray', 'cyan')
    cmap = ListedColormap(colors[:len(np.unique(y))])
#     plot the decision surface
    x1_min, x1_max = X[:, 0].min() - 1, X[:, 0].max() + 1
    x2_min, x2_max = X[:, 1].min() - 1, X[:, 1].max() + 1
    xx1, xx2 = np.meshgrid(np.arange(x1_min, x1_max, resolution),
                          np.arange(x2_min, x2_max, resolution))
    Z = classifier.predict(np.array([xx1.ravel(), xx2.ravel()]).T)
    Z = Z.reshape(xx1.shape)
    plt.contourf(xx1, xx2, Z, alpha=0.3, cmap=cmap)
    plt.xlim(x1_min, x1_max)
    plt.ylim(x2_min, x2_max)

    for idx, cl in enumerate(np.unique(y)):
        plt.scatter(
            X[y==cl, 0], X[y==cl, 1],
            alpha=0.8,
            c=colors[idx],
            marker=markers[idx],
            label=cl,
            edgecolor='black'
        )
        
#         highlight test samples
    if test_idx:
#         plot all samples
        X_test, y_test = X[test_idx, :], y[test_idx]
        plt.scatter(
            X_test[:, 0], X_test[:, 1],
            c='',
            edgecolor='black',
            alpha=1.0,
            label='test set'
        )

def get_player_data(data_path='players.csv', normalized=True):
    # Load players data
    player_df = pd.read_csv(data_path, index_col=False)

    # Filter the data by `Pos` of G and C
    pos_dict = {'G':1, 'C':-1}
    mask = player_df['Pos'].apply(lambda x: x in pos_dict)
    player_df = player_df[mask]

    height = player_df['Ht'].values
    weight = player_df['Wt'].values
    X_player = np.stack([height, weight], axis=1)
    y_player = player_df['Pos'].map(pos_dict)

    # Normalization
    if normalized:
        player_mean = X_player.mean(axis=0)
        player_std = X_player.std(axis=0)
        X_player_n = (X_player - player_mean) / player_std

        return X_player_n, y_player
    return X_player, y_player