# Financial-Transaction-Category-Classifier-Model
Machine Learning algorithm using the Random Forest Classifier to predict 'Category' field in Financial Transaction data.  ECDF visual program provided to gain additional insight on data before starting the modeling process.

Program uses the following packages: os, pandas, datetime, numpy, sklearn

1. Split using ALL data in transaction table to categorize 'Category' field in data
2. Create the basic token pattern and obtain text & numberic data
3. Instantiate nested pipeline
4. Use GridSearchCV to find parameters which result in highest accuracy score
5. Fit to the training data then compute accuracy
6. Creates Excel spreadsheet with predictions from model in specified folder
