"""Build a sentiment analysis / polarity model

Sentiment analysis can be casted as a binary text classification problem,
that is fitting a linear classifier on features extracted from the text
of the user messages so as to guess wether the opinion of the author is
positive or negative.

In this examples we will use a movie review dataset.

"""
# Author: Olivier Grisel <olivier.grisel@ensta.org>
# License: Simplified BSD

import sys
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.svm import LinearSVC
from sklearn.pipeline import Pipeline
from sklearn.model_selection import GridSearchCV
from sklearn.datasets import load_files
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn import metrics
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import RandomForestClassifier, AdaBoostClassifier
from sklearn import svm
from sklearn.neural_network import MLPClassifier

if __name__ == "__main__":
    # NOTE: we put the following in a 'if __name__ == "__main__"' protected
    # block to be able to use a multi-core grid search that also works under
    # Windows, see: http://docs.python.org/library/multiprocessing.html#windows
    # The multiprocessing module is used as the backend of joblib.Parallel
    # that is used when n_jobs != 1 in GridSearchCV

    # the training data folder must be passed as first argument
    movie_reviews_data_folder = r"./data"
    dataset = load_files(movie_reviews_data_folder, shuffle=False)
    print("n_samples: %d" % len(dataset.data))

    # split the dataset in training and test set:
    docs_train, docs_test, y_train, y_test = train_test_split(
        dataset.data, dataset.target, test_size=0.25, random_state=None)

    # TASK: Build a vectorizer / classifier pipeline that filters out tokens
    # that are too rare or too frequent

    pipe = Pipeline([
        ('tfidf', TfidfVectorizer(min_df=0.05, max_df=0.95)),
        ('clf', LinearSVC(C=1000))
        # ('clf', svm.SVC(random_state=45))
        # ('clf', LogisticRegression(random_state=42))
        # ('random_forest', RandomForestClassifier(
        #     n_estimators=200, random_state=0))
        # ('clf', MLPClassifier(alpha=1, max_iter=1000))
        # ('abc', AdaBoostClassifier(n_estimators=50,
        #                          learning_rate=1))
    ])

    # TASK: Build a grid search to find out whether unigrams or bigrams are
    # more useful.
    # Fit the pipeline on the training set using grid search for the parameters
    parameters = {
        'tfidf__ngram_range': [(1, 1), (1, 2)],
    }
    grid_search = GridSearchCV(estimator=pipe,
                               param_grid=parameters,
                               scoring='accuracy',
                               cv=10)
    grid_search.fit(docs_train, y_train)

    # TASK: print the cross-validated scores for the each parameters set
    # explored by the grid search

    n_candidates = len(grid_search.cv_results_['params'])
    for i in range(n_candidates):
        print(i, 'params - %s; mean - %0.2f; std - %0.2f'
              % (grid_search.cv_results_['params'][i],
                 grid_search.cv_results_['mean_test_score'][i],
                 grid_search.cv_results_['std_test_score'][i]))

    # TASK: Predict the outcome on the testing set and store it in a variable
    # named y_predicted

    pipe.fit(docs_train, y_train)
    y_predicted = pipe.predict(docs_test)

    # Print the classification report
    print(metrics.classification_report(y_test, y_predicted,
                                        target_names=dataset.target_names))

    # Print and plot the confusion matrix
    cm = metrics.confusion_matrix(y_test, y_predicted)
    print(cm)

    # import matplotlib.pyplot as plt
    # plt.matshow(cm)
    # plt.show()
