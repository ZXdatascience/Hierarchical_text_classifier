from vectorize import vectorize
from label_tree import Tree, RestoredTree
from evaluate import evaluate

from sklearn.model_selection import train_test_split
from sklearn.datasets import fetch_20newsgroups

############################
# Variables
############################
saved_model_location = 'model/'
classifier_type = 'SVM'
max_level = 6
test_ratio = 0.2

############################
# Get data
############################
newsgroups_train = fetch_20newsgroups(subset='train')
newsgroups_test = fetch_20newsgroups(subset='test')
X_train = newsgroups_train.data
y_train = [l.split('/')[-2] for l in newsgroups_train.filenames]
y_train = [['Top/' + '/'.join(label.split('.'))] for label in y_train]

X_test = newsgroups_test.data
y_test = [l.split('/')[-2] for l in newsgroups_test.filenames]
y_test = [['Top/' + '/'.join(label.split('.'))] for label in y_test]

vectorizer, X_train_matrix = vectorize(X_train)

#############################
# Build the tree, train it and evaluate prediction
#############################
tree = Tree(classifier_type, X_train_matrix, y_train, max_level)
tree.train()
predict_labels = tree.predict_text(X_test, vectorizer)
eval_res = evaluate(y_test, predict_labels, 6)

print('precision:')
print(eval_res[0])
print('recall:')
print(eval_res[1])
print('F1:')
print(eval_res[2])

##############################
# Save the model
##############################
tree.save_model(saved_model_location)

##############################
# Restore the model from saved model location
##############################
test_text = X_test[:4]
restored_tree = RestoredTree(saved_model_location)
result = restored_tree.predict_text(test_text, vectorizer)
# It is important that the vectorizer used here must be the vectorizer used in vectorizing train text.
print(result)