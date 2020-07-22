#! python3
"""
Created on Sun Nov 10 11:02:32 2019

Multi Data Type Pipeline using RandomForestClassifier to predict 'Category' 
field in Mint Transaction - Daily data

Note: OneVsRestClassifier with LogisticRegression & KNeighborsClassifier 
    yields lower accuracy score & Account Name column lowers accuracy score
    
Steps:
    1. Import modules
    2. Import dataset and split training & testing data
    3. Build Pipeline to Impute numeric data and Count Vectorize text data
    4. Create GridSearchCV to find best hyperparamters and fit training data to model
    5. Print model performance
    6. Create excel spreadsheet with model's predictions for y target variable (Category)

@author: HM
"""

""" #NOTE: SparseInteraction function below not used in program b/c it slows down
#computation
#from itertools import combinations
#
#import numpy as np
#from scipy import sparse
#from sklearn.base import BaseEstimator, TransformerMixin
#
#class SparseInteractions(BaseEstimator, TransformerMixin):
#    def __init__(self, degree=2, feature_name_separator="_"):
#        self.degree = degree
#        self.feature_name_separator = feature_name_separator
#
#    def fit(self, X, y=None):
#        return self
#
#    def transform(self, X):
#        if not sparse.isspmatrix_csc(X):
#            X = sparse.csc_matrix(X)
#
#        if hasattr(X, "columns"):
#            self.orig_col_names = X.columns
#        else:
#            self.orig_col_names = np.array([str(i) for i in range(X.shape[1])])
#
#        spi = self._create_sparse_interactions(X)
#        return spi
#
#    def get_feature_names(self):
#        return self.feature_names
#
#    def _create_sparse_interactions(self, X):
#        out_mat = []
#        self.feature_names = self.orig_col_names.tolist()
#
#        for sub_degree in range(2, self.degree + 1):
#            for col_ixs in combinations(range(X.shape[1]), sub_degree):
#                # add name for new column
#                name = self.feature_name_separator.join(self.orig_col_names[list(col_ixs)])
#                self.feature_names.append(name)
#
#                # get column multiplications value
#                out = X[:, col_ixs[0]]
#                for j in col_ixs[1:]:
#                    out = out.multiply(X[:, j])
#
#                out_mat.append(out)
#
#        return sparse.hstack([X] + out_mat)"""

# Import necessary modules
import os, pandas as pd, datetime, numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV
from sklearn.preprocessing import Imputer, FunctionTransformer, StandardScaler
from sklearn.metrics import classification_report
from sklearn.pipeline import Pipeline, FeatureUnion
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.ensemble import RandomForestClassifier

# Ignores Warnings
import warnings
warnings.filterwarnings(action='ignore')

# Changes current working directory to where Transaction file is stored
os.chdir('C:\\Users\\User\\Documents\\Python Scripts\\Machine Learning\\Mint\\Data\\')

table = pd.read_csv('Mint Transactions - Daily.csv')

X_data = table[['$ Amount', 'Description']]
Y_data = table['Category']

# Split using ALL data in transaction table to categorize 'Category' field in data
X_train, X_test, y_train, y_test = train_test_split(X_data,
                                                    Y_data,
                                                    test_size=0.2,
                                                    random_state=22)


# Create the basic token pattern
# NOTE: Text tokens in vectorizer below decreases accuracy score
TOKENS_BASIC = '\\S+(?=\\s+)'

# Obtain the text data: get_text_data
get_text_data1 = FunctionTransformer(lambda x: x['Description'], validate=False)

# Obtain the numeric data: get_numeric_data
get_numeric_data = FunctionTransformer(lambda x: x[['$ Amount']], validate=False)

# Create a FeatureUnion with nested pipeline: process_and_join_features
# NOTE: ngram_range used to go up to 4 n grams for vectorizing text
process_and_join_features = FeatureUnion(
            transformer_list = [
                ('numeric_features', Pipeline([
                    ('selector', get_numeric_data),
                    ('imputer', Imputer())
                ])),
                ('text_features1', Pipeline([
                    ('selector', get_text_data1),
                    ('vectorizer', CountVectorizer(TOKENS_BASIC, ngram_range=(1,4)))
                ]))
            ]
        )

# Instantiate nested pipeline: pl
pl = Pipeline([
        ('union', process_and_join_features),
#        NOTE: SparseInteractions above slows down model computation
#        ('int', SparseInteractions(degree=2)),
#       NOTE: Standard Scalar imported to reduce large variances in data inputs
        ('scaler', StandardScaler(with_mean=False)),
        ('clf', RandomForestClassifier())
    ])

# Create parameters variable for GridSearchCV function to be used
parameters = {'clf__n_estimators':np.arange(1,5)}

# Use GridSearchCV to find parameters which make model most accurate
cv = GridSearchCV(pl, param_grid=parameters)

# Fit cv to the training data
cv.fit(X_data, Y_data)
y_pred = cv.predict(X_test)

# Compute and print accuracy
accuracy = cv.score(X_test, y_test)
print("\nAccuracy on test data: {:.1%}\n".format(accuracy))

# Calculates cross-validation score to compute numpy mean and prints result
cv_scores = cross_val_score(pl, X_data, Y_data, cv=10)
print('\nCV scores: {}'.format(cv_scores))
print('\nCV score mean: {:.1%}'.format(np.mean(cv_scores)))
print('\n')

# Prints best parameters and classifcation report by each Category within Target Variable
print(cv.best_params_)
print('\n')
print(classification_report(y_test, y_pred))



# --** CREATES EXCEL SPREADSHEET WITH PREDICTIONS FROM MODEL **--


# Creates Pandas DataFrame of test data with model prediction column
results = pd.DataFrame(X_test)
results['Prediction'] = cv.predict(X_test)

# Joins correct 'Category' data from Y_data var to compare against model prediction
results = results.join(Y_data)

# Creates new Category Classifier Model Prediction spreadsheet and puts in folder
date = datetime.datetime.today().strftime('%m-%d-%Y')

file_name = 'C:\\Users\\User\\Documents\\Python Scripts\\Machine Learning\\' \
            'Prediction History\\Category Classifier Prediction History\\' + date + \
            ' Category Classifier Model Prediction.xlsx'

try:
    results.to_excel(file_name)
except PermissionError:
    print('\nFile for ' + date + ' has already been created')