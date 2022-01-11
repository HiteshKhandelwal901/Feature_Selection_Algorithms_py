from skmultilearn.adapt import MLkNN
from scipy import sparse
from skmultilearn.dataset import load_dataset
import sklearn.metrics as metrics
from sklearn.metrics import hamming_loss


import warnings
warnings.filterwarnings("ignore")


X_train, y_train, feature_names, label_names = load_dataset('emotions', 'train')
X_test, y_test, _, _ = load_dataset('emotions', 'test')

clf = MLkNN(k=5)

clf.fit(X_train, y_train)

y_pred = clf.predict(X_test)

print(type(y_pred))  # <class 'scipy.sparse.lil.lil_matrix'>

print(type(y_pred.toarray()))  # <class 'numpy.ndarray'>

y_pred_csr = sparse.csr_matrix(y_pred).toarray()

print(type(y_pred_csr))  # <class 'scipy.sparse.csr.csr_matrix'>

accuracy = metrics.accuracy_score(y_test, y_pred)
print(accuracy)  # 0.148

loss = hamming_loss(y_pred, y_test)

print("loss = ", loss)