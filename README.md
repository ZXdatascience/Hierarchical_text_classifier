# Hierarchical_text_classifier
## Fast text classification for data with hierarchical labels
**Intro**:

This is a multi-output classifier that can boost the training for data with hierarchical labels: eg.Top/News/Sports. Suppose there are N training samples with k unique labels, the complexity of this classifier with SVM as base classifier is O(nlogk) while a SVM has time complexity of O(nk). The more number of unique labels, the more time you save on training.


1. It can be used for both text and non-text data.

2. You can change or add the base classifier you want to use in base_classifier.py

Please see example.py for instructions
