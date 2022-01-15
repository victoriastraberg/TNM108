import sklearn
import numpy as np
from sklearn import metrics 
from sklearn.datasets import load_files
from sklearn.feature_extraction.text import CountVectorizer 
from sklearn.feature_extraction.text import TfidfTransformer 
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline
from sklearn.linear_model import SGDClassifier
from sklearn.model_selection import GridSearchCV 

moviedir = r'movie_reviews'

# loading all files
movie = load_files(moviedir, shuffle=True)

print(len(movie.data))

# target names ("classes") are automatically generated from subfolder names
print(movie.target_names)

# Split data into training and test sets
from sklearn.model_selection import train_test_split
docs_train, docs_test, y_train, y_test = train_test_split(movie.data, movie.target, test_size = 0.5, random_state = 12)

# Building a pipline 
text_clf = Pipeline([
('vect', CountVectorizer()),
('tfidf', TfidfTransformer()),
('clf', SGDClassifier(loss='hinge', penalty='l2',alpha=1e-3, random_state=42
,max_iter=5, tol=None)),
])
# Train the model 
text_clf.fit(docs_train, y_train)
# training SVM classifier
predicted = text_clf.predict(docs_test)
print("SVM accuracy ",np.mean(predicted == y_train))

print(metrics.classification_report(y_train, predicted,target_names=movie.target_names))
# Confusion matrix 
print(metrics.confusion_matrix(y_train, predicted))

# Parameter tuning using grid search
parameters = {
    'vect__ngram_range': [(1, 1), (1, 2)],
    'tfidf__use_idf': (True, False),
    'clf__alpha': (1e-2, 1e-3),
}

gs_clf = GridSearchCV(text_clf, parameters, cv=5, n_jobs=-1)
# Fitting the Grid-Search to training data 
gs_clf = gs_clf.fit(docs_train[:600], y_train[:600])

# very short and fake movie reviews
reviews_new = ['This movie was excellent', 'Absolute joy ride', 
            'Steven Seagal was terrible', 'Steven Seagal shone through.', 
              'This was certainly a movie', 'Two thumbs up', 'I fell asleep halfway through', 
              "We can't wait for the sequel!!", '!', '?', 'I cannot recommend this highly enough', 
              'instant classic.', 'Steven Seagal was amazing. His performance was Oscar-worthy.']

# Get the best score and params for one review 
#print(movie.target_names[gs_clf.predict([reviews_new])[0]])

print('\nBest score:',gs_clf.best_score_,'\n')

for param_name in sorted(parameters.keys()):
    print("%s: %r" % (param_name, gs_clf.best_params_[param_name]))

# Call predict on the estimator with the best found parameters
pred = gs_clf.predict(reviews_new)

# Print results
print('\n')
for review, category in zip(reviews_new, pred):
    print('%r => %s' % (review, movie.target_names[category]))
