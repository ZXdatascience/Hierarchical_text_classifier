from sklearn.svm import LinearSVC
from sklearn.linear_model import LogisticRegression
from sklearn.multioutput import MultiOutputClassifier

def base_classifier(tree_type):
    """
    You can add your own base classifiers here. To make sure it works, your selection of base classifier must be compatible
    with the multioutput classifier sklearn provides.
    :param tree_type: String
    :return:
    """
    if tree_type == 'LR':
        classifier = MultiOutputClassifier(LogisticRegression())
    if tree_type == 'SVM':
        classifier = MultiOutputClassifier(LinearSVC())
    return classifier