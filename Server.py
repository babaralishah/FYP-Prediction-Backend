# Import Libraries :
# Calculating the processing time
import time
from flask import Flask, request, jsonify
import os
from flask import Flask, jsonify, render_template
from flask import make_response, url_for, Blueprint
from sklearn.naive_bayes import GaussianNB, MultinomialNB
from sklearn.svm import SVC, NuSVC
from Test import test
from flask import make_response, url_for
from sklearn import metrics
import matplotlib.pyplot as plt
from flask import Flask, render_template, request, redirect
import numpy as np
from flask_cors import CORS, cross_origin
import pandas as pad
from werkzeug.utils import secure_filename
import csv
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier, GradientBoostingClassifier
from sklearn.tree import DecisionTreeClassifier, ExtraTreeClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.preprocessing import LabelEncoder
from sklearn import naive_bayes
from sklearn.preprocessing import MinMaxScaler
from sklearn.feature_selection import chi2
from sklearn.feature_selection import SelectKBest
from sklearn.linear_model import LinearRegression
from sklearn.metrics import accuracy_score
import pandas as pd
from flask import jsonify
import warnings
from string import Template
warnings.filterwarnings("ignore", category=FutureWarning)

# import various ML algorithms to be used from the library
# Naive Bayes Classifier

# SVM

# LabelEncoder

# KNN

# Splitting

# NB

# Logistic Regression

# DTClassifier

# RFC
# from flask_restful import Resource, Api;
# Required Imports
# from firebase_admin import credentials, firestore, initialize_app

app = Flask(__name__)

# firebase = firebase.FirebaseApplication('https://testwhatsapp-pxqrtl.firebaseio.com/', None)

# cors = CORS(app, resources={r"/api/http://localhost:4200": {"origins": "http://localhost:4200"}})# api = Api(app)
CORS(app, supports_credentials=True)
app.register_blueprint(test, url_prefix="/")

# Defining the global variables
X = 0
y = 0
X_data = 0
y_data = 0
data_frame1 = 0

# #########################################################################################################################


@app.route('/')
def index():
    return 'abcdef'

# #########################################################################################################################


@app.route('/output1/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def upload_file1():
    global X

    columns = list(X.columns.values)
    return jsonify(columns)


# ##########################################################################################################################


@app.route('/output2/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def upload_file2():
    col_name = request.data
    col_name = col_name.decode('utf-8')
    print('Calling the uplaod file2: ')
    # print(col_name)
    global data_frame1
    # print(data_frame1[col_name])
    y_data = data_frame1.iloc[:, [-1]]
    y_data = y_data.to_json()
    abc = ' '
    abc = data_frame1[col_name].value_counts()
    # print('\n\n\n')
    # print(abc)
    # print('\n\n\n')
    a = "Hello"
    # print('\nbefore\n')
    # print(type(abc))
    abc = abc.to_string()
    # print('\nafter\n')
    # print(type(abc))
    # print(abc)
    # print('\n\n\n')
    return jsonify(abc, a)

# ###########################################################################################################################


@app.route('/output/', methods=['POST', 'GET'])
@cross_origin(allow_headers=['http://localhost:4200'])
def upload_file():

    if request.method == 'POST':
        file = request.files['file']
        file.save(secure_filename(file.filename))
        print('File received at rest service: ' + file.filename)
        # Data Preprocessing:
        # print('Calling the uplaod file function in the same file')
        # upload_file1(file)
        # Reading Data file (csv) from web cache
        # global data_frame1
        # data_frame1 = pad.read_csv(file.filename)
        print('step-01')
        data_frame = pad.read_csv(file.filename)

        # Removing all columns with only one value, or have more than 50% missing values to work faster
        # data_frame=data_frame.dropna(axis=1)

        # Replace nan values with average of columns

        # Start calculating the time
        start = time.time()
        print('\nnans before removing\n')
        nan = data_frame.isnull().sum().sum()
        print(nan)
        global data_frame1
        data_frame1 = pad.read_csv(file.filename)

        if nan != 0:
            data_frame1 = data_frame1.fillna(data_frame1.mean())
            data_frame = data_frame.fillna(data_frame.mean())

        print('\nnans after removing\n')
        print(data_frame.isnull().sum().sum())
        # Stops the watch
        end = time.time()
        print('\n\nPerformed nan replacement to  data_frame, and data_frame1 in time ',
              end-start, '\n\n')

        from sklearn.preprocessing import OneHotEncoder, LabelEncoder
        # ohe = OneHotEncoder()
        le = LabelEncoder()

        # Converting columns from object dtype to int if there is any in the dataframe
        for x in data_frame.columns:
            if data_frame[x].dtype == 'object':
                data_frame[x] = le.fit_transform(data_frame[x])

        # Replace negative numbers in Pandas Data Frame by zero
        data_frame[data_frame < 0] = 0

        # Convert Floats to ints
        for x in data_frame.columns:
            data_frame[x] = data_frame[x].astype(int)

        # All rows and columns (Attributes/Features) except last column
        global X
        X = data_frame.iloc[:, :-1]

        # Only Last row and column (Label)
        global y
        y = data_frame.iloc[:, [-1]]

        # Train-test split
        a = len(data_frame.index)
        # Train-test split
        if a >= 1 and a <= 1000:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.30, train_size=0.70, random_state=0)
            print('1')
        elif a >= 1000 and a <= 10000:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.07, train_size=0.03, random_state=0)
            print('2')
        elif a >= 10000 and a <= 100000:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.007, train_size=0.003, random_state=0)
            print('3')
        else:
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, test_size=0.0007, train_size=0.0003, random_state=0)

        # print(X_train.shape)
        # k = X_train.columns()/2

        # Standard scaling of the dataset
        from sklearn.preprocessing import StandardScaler

        sc = StandardScaler()
        X_train = sc.fit_transform(X_train)
        X_test = sc.transform(X_test)

        # Principle Component Analysis  (Feature Selection)
        from sklearn.decomposition import PCA

        pca = PCA(n_components=2)
        X_train = pca.fit_transform(X_train)
        X_test = pca.transform(X_test)

        # Applying Univariate Selection  (Feature Selection)
        # (Most appropriate Columns among all columns)
        # num_feats = 5
        #X_train = SelectKBest(chi2,k=num_feats).fit_transform(X_train, y_train)
        #X_test = SelectKBest(chi2, k=10).fit_transform(X_test, y_test)

        # print('\n\n\nX_train\n\n')
        # print(X_train.shape)

        # ML Models Building:

        # Predict Linear Regression
        #y_pred = linear.predict(X_test)

        # Models Declaration and Definition

        # Linear Regression
        linear = LinearRegression()

        # LogisticRegression
        LR_clf = LogisticRegression()

        # DecisionTreeClassifier
        DT_clf = DecisionTreeClassifier()

        # KNeighborsClassifier
        knn_clf = KNeighborsClassifier(n_neighbors=1)

        # Support Vector Machine
        svc_clf = SVC()

        # naive_bayes
        nbclf = naive_bayes.GaussianNB()

        # RandomForestClassifier
        RF_clf = RandomForestClassifier(random_state=0)

        # Feed the Model:

        # Training of Models on Train set

        # fit the model with data
        linear.fit(X_train, y_train)

        # Calculating the processing time
        # import time

        start = time.time()

        # fit the model with data
        LR_clf.fit(X_train, y_train)  # .values.ravel())

        # Stops the watch
        end = time.time()
        # Calculates the consumed time
        # print("\nExecution time of Linear regression training: ", end - start)

        # Start calculating the time
        start = time.time()

        # Stops the watch
        end = time.time()

        # Calculates the consumed time
        # print("\nExecution time of Logistic regression training: ", end - start)

        start = time.time()

        # fit the model with data
        DT_clf.fit(X_train, y_train)
        end = time.time()
        # print("\nExecution time of Decision Tree training: ", end - start)

        start = time.time()

        # fit the model with data
        knn_clf.fit(X_train, y_train.values.ravel())

        # Stops the watch
        end = time.time()

        # Calculates the consumed time
        # print("\nExecution time of Knn training: ", end - start)

        # fit the model with data
        svc_clf.fit(X_train, y_train.values.ravel())
        end = time.time()
        # print("\nExecution time of SVC training: ", end - start)

        start = time.time()

        # fit the model with data
        nbclf.fit(X_train, y_train.values.ravel())

        # Stops the watch
        end = time.time()

        # Calculates the consumed time
        # print("\nExecution time of Naive Bayes training: ", end - start)

        start = time.time()

        # fit the model with data
        RF_clf.fit(X_train, y_train.values.ravel())

        # Stops the watch
        end = time.time()

        # Calculates the consumed time
        # print("\nExecution time of Random Forest training: ", end - start)

        print("\n\nModel \t\t\t\t\t Test Score\t\t\t\t\tTrain Score")

        # Start calculating the time
        start = time.time()

        # Model Validation:

        # Predict Linear Regression
        # y_pred = linear.predict(X_test)

        # Accuracy of Logistic Regression
        print("\nLinear Regression Accuracy Score:\t", linear.score(
            X_test, y_test)*100, "\t\t\t\t", linear.score(X_train, y_train)*100)
        linear = linear.score(X_test, y_test)*100
        # Stops the watch
        end = time.time()

        # Calculates the consumed time
        #print("\nExecution time of Linear regression Predicting: ",end - start)

        # Start calculating the time
        start = time.time()

        # Accuracy of Logistic Regression
        print("\nLogistic Regression Accuracy Score:\t", LR_clf.score(
            X_test, y_test)*100, "\t\t\t\t", LR_clf.score(X_train, y_train)*100)
        lr = LR_clf.score(X_test, y_test)*100
        # Stops the watch
        end = time.time()

        # Calculates the consumed time
        #print("\nExecution time of Logistic regression Predicting: ",end - start)

        # Start calculating the time
        start = time.time()

        # Accuracy of Decision Tree
        print("\nDecision Tree Accuracy Score:\t\t", DT_clf.score(
            X_test, y_test)*100, "\t\t\t\t", DT_clf.score(X_train, y_train)*100)
        dt = DT_clf.score(X_test, y_test)*100
        # Stops the watch
        end = time.time()
        # Calculates the consumed time
        #print("\nExecution time of Decision Tree Predicting: ",end - start)

        # Start calculating the time
        start = time.time()

        # Accuracy of K nearest Neighbours (KNN)
        print("\nK nearest Neighbours Accuracy Score:\t", knn_clf.score(
            X_test, y_test)*100, "\t\t\t\t", knn_clf.score(X_train, y_train)*100)
        end = time.time()
        knn = knn_clf.score(X_test, y_test)*100
        # Calculates the consumed time
        #print("\nExecution time of Knn Predicting: ",end - start)

        # Start calculating the time
        start = time.time()

        # Accuracy of Rabdom Forest
        print("\nRandom Forest Accuracy Score:\t\t", RF_clf.score(
            X_test, y_test)*100, "\t\t\t\t", RF_clf.score(X_train, y_train)*100)
        # rf = RF_clf.score(X_test, y_test)*100
        # Stops the watch
        end = time.time()

        # Calculates the consumed time
        #print("\nExecution time of Random Forest Predicting: ",end - start)

        # Start calculating the time
        start = time.time()

        # Accuracy of Naive Bayes Gaussian
        print("\nNaive Bayes Gaussian Accuracy Score:\t", nbclf.score(
            X_test, y_test)*100, "\t\t\t\t", nbclf.score(X_test, y_test)*100)
        nbclf = nbclf.score(X_test, y_test)*100
        # Stops the watch
        end = time.time()
        #global svc
        # Accuracy of Naive Bayes Gaussian
        # print("\nSupport Vector Machine Accuracy Score:\t",svc_clf.score(X_test, y_test)*100,"\t\t\t\t",nbclf.score(X_test,y_test)*100)

        # svc = svc_clf.score(X_test, y_test)*100
        #abc = 'hello'

        results = {
            "data": [
                {
                    "Algorithm": "Linear Regression",
                    "Efficiency": linear
                },  {
                    "Algorithm": "Logistic Regression",
                    "Efficiency": lr
                }, {
                    "Algorithm": "Decision Tree",
                    "Efficiency": dt
                },  {
                    "Algorithm": "K nearest neighbor",
                    "Efficiency": knn
                },  {
                    "Algorithm": "Naive Bayes",
                    "Efficiency": nbclf
                }, {
                    "Algorithm": "Support Vector",
                    "Efficiency": '  '
                }

            ]
        }
        return jsonify([results])


if __name__ == '__main__':
    app.run(debug=True)
