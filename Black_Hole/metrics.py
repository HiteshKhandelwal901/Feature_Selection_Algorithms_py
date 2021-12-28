from sklearn.metrics import f1_score
import tensorflow as tf


y_true = [[0, 0, 0], [1, 1, 1], [0, 1, 1]]
y_pred = [[0, 0, 0], [1, 1, 1], [1, 1, 0]]

s = f1_score(y_true, y_pred, average= 'macro')

print("s = ", s)

def f1loss(y_true, y_pred, loss = "macro"):
    if loss == "macro":
        return f1_score(y_true, y_pred, average= "macro")
    if loss == "micro":
        return f1_score(y_true, y_pred, average= "micro")
    if loss == "weighted":
        return f1_score(y_true, y_pred, average= "weighted")

