import numpy as np
from sklearn.metrics import label_ranking_average_precision_score


if __name__ == '__main__':
    y_true = np.array([
        [1,0,0,0,1],
    ])

    y_pred = np.array([
        [0, 0, 0, 0, 0],
    ])

    assert y_true.shape == y_pred.shape

    metric = label_ranking_average_precision_score(y_true,y_pred)
    print(metric)