import streamlit as st
from sklearn import datasets
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt

st.title("Streamlit example")

st.write("""
This is my first streamlit app.

I am following the following sources:

[Python Engineer](https://www.youtube.com/watch?v=Klqn--Mu2pE&t=351s)
[Data Professor](https://www.youtube.com/watch?v=ZZ4B0QUHuNc)

""")

st.write('# Analysis')

dataset_name = st.sidebar.selectbox("Select Dataset", ("Iris", 'Breast Cancer', 'Wine'))

st.write("Working with the {} dataset".format(dataset_name))

clf_name = st.sidebar.selectbox("Select Classifier", ("KNN", "SVM", "Random Forest"))



def get_dataset(dataset_name):
    if dataset_name == "Iris":
        data = datasets.load_iris()
    elif dataset_name == "Breast Cancer":
        data = datasets.load_breast_cancer()
    else:
        data = datasets.load_wine()

    X = data.data
    y = data.target

    return X,y



X, y = get_dataset(dataset_name)

st.write("Shape: ", X.shape)
st.write("Number of classes: ", len(np.unique(y)))

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size = 0.3, random_state = 1234)


def add_parameter_ui(clf_name):
    params = {}
    if clf_name == "KNN":
        K = st.sidebar.slider("K", 1, 15)
        params['K'] = K
    elif clf_name == "SVM":
        C = st.sidebar.slider("C", 0.01, 10.0)
        params['C'] = C
    else:
        max_depth = st.sidebar.slider('Max depth', 2, 15)
        n_estimators = st.sidebar.slider('Number of estimators', 2, 15)
        params['max_depth'] = max_depth
        params['n_estimators'] = n_estimators

    return params


params = add_parameter_ui(clf_name)


def get_clf(clf_name, params):
    if clf_name == "KNN":
        clf = KNeighborsClassifier(n_neighbors = params['K'])
    elif clf_name == "SVM":
        clf = SVC(C = params['C'])
    else:
        clf = RandomForestClassifier(n_estimators = params['n_estimators'],
                                    max_depth = params['max_depth'],
                                    random_state = 1234)
    return clf

model = get_clf(clf_name, params)

model.fit(X_train, y_train)

y_pred = model.predict(X_test)

acc = accuracy_score(y_test, y_pred)

acc = round(acc, 3)

st.write("{} accuracy {}".format(clf_name, acc))

### PLOT
pca = PCA(2)
x_projected = pca.fit_transform(X)
x1 = x_projected[:, 0]
x2 = x_projected[:, 1]

fig = plt.figure()
plt.scatter(x1, x2, c=y, alpha=0.8, cmap='viridis')
plt.xlabel('PC1')
plt.ylabel('PC2')
plt.colorbar()

st.pyplot(fig)
