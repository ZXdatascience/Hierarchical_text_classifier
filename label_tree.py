from base_classifier import base_classifier
from sklearn.preprocessing import MultiLabelBinarizer
import numpy as np
from sklearn.exceptions import NotFittedError
from sklearn.externals import joblib
import pickle

class Node:
    """
    A node is a basic component of the tree that has a classifier in it with other supplemental information.
    """
    def __init__(self, key, inp, l):
        """

        :param key: String; The name of the node.
        :param inp: list; A list of index of the training examples.
        :param l: int; The level of the node. eg. Top/Sports/Baseball, Baseball is in level 2. Top in level 0.

        """
        self.key = key
        self.classifier = None
        # the classifier of the node, will be initialized in training.
        self.input = inp
        # It is a list of index of training examples. While training, the input of this node's classifier will be extracted
        # from the training matrix using these indices. Because each layer of the tree will need exactly the whole data set
        # for training, we can save a lot of space by using indices.
        self.labels = []
        # A list of string labels of training data. eg. ['Top/News/Politics', 'Top/Sports/Basketball']
        self.child = {}
        # A dictionary of child nodes. Key of this dictionary will be the child nodes'names. Values are corresponding Node
        # object.
        self.pre_node = None
        # This node's parent Node.
        self.score = 0
        self.level = l
        # This node's level.
        self.mlb = MultiLabelBinarizer()
        # This transfers the multi-label of a datum into binary label.
        self.ignore_labels = None


class Tree:
    def __init__(self, classifier_type, matrix, labels, lev):
        """
        :param classifier_type: SVM, RL, or any other binary classifier
        :param matrix: the train data matrix
        :param labels: the labels for the train data eg. ['Top/News/Sports', 'Top/News/Politics']
        :param lev: the maximum height of the tree.
        """
        print("Building the tree")
        self.root = Node("Top", [], 0)
        self.root.score = 1
        self.tree_type = classifier_type
        self.matrix = matrix
        cur = self.root
        for i, labels_per_example in enumerate(labels):
            if not labels_per_example:
                continue
            for label in labels_per_example:
                levels = label.split("/")[1:]
                # ignore the first "Top" taxonomy
                for j, key in enumerate(levels):
                    if j >= lev:
                        continue
                    # only consider nodes below level lev
                    child_keys = [c for c in cur.child.keys()]
                    if key not in child_keys:
                        new_node = Node(key, [], cur.level + 1)
                        cur.child[key] = new_node
                        new_node.pre_node = cur
                        if cur.input and cur.input[-1] == i:
                            cur.labels[-1].add(key)
                        else:
                            cur.input.append(i)
                            cur.labels.append({key})
                        cur = cur.child[key]
                    else:
                        if cur.input and cur.input[-1] == i:
                            cur.labels[-1].add(key)
                        else:
                            cur.input.append(i)
                            cur.labels.append({key})
                        cur = cur.child[key]
                cur = self.root
            if i % 200000 == 0:
                print("{0} documents are processed".format(i))

    def get_model_structure(self, levels):
        """
        This is a helper function to print the tree's structure.
        It returns a list of list of Strings. [[node names for level 0], [node names for level 1], [...]]
        You can check the nodes name in each level.
        :param levels: list[int] The levels you may wanna check of the tree.
        :return: [[String]]
        """
        queue = [self.root]
        node_names = []
        levels.sort()
        l = 0
        levels_index = 0
        while queue and l <= levels[-1]:
            # iterate to the last number of levels.
            temp = []
            if l == levels[levels_index]:
                node_names.append([node.key for node in queue])
                print("%s th level: %s nodes"%(l+1, len(queue)))
                levels_index += 1
            while queue:
                node = queue.pop(0)
                # dequeue
                temp.extend(node.child.values())
            queue = temp
            l += 1
        return node_names

    def traverse(self):
        """
        The level order traversal of the nodes in the tree.
        :return: List.
        """
        queue = [self.root]
        traverse_list = []
        while queue:
            node = queue.pop(0)
            # dequeue
            traverse_list.append(node)
            queue.extend(node.child.values())
        return traverse_list

    def train(self):
        """
        the training process is:
        1. traverse of the tree
        2. train the model in each Node, where the train data is in node.input (list of input index)
           and the labels are already stored in the node.labels while building the tree.
        :return: None
        """
        print("start training!!")
        traverse_list = self.traverse()
        self.train_model(traverse_list)

    def train_model(self, traverse_list):
        """
        The model is trained by training the base classifier in each node in a level-order.
        :param traverse_list: level order traversal of the tree.
        :return: None
        """
        for i, node in enumerate(traverse_list):
            if i % 100 == 0:
                print("Trained {0} models".format(i))
            if len(node.child.keys()) <= 1:
                continue
            # if there is only one child for this node, no need to train, continue.
            if not node.child:
                continue
            train_data = self.matrix[node.input]
            train_labels = node.mlb.fit_transform(node.labels)
            node.classifier = base_classifier(self.tree_type)

            try:
                node.classifier.fit(train_data, train_labels)
            # This try except block is to handle the case that one child label exists in all the inputs of this node.
            # for ex, suppose there are only two input of the node Health. ['Health/Diseases/Flu', 'Health/Public health']
            # for the first input. ['Health/Diseases/Diabetes', 'Health/Obesity'] for the 2nd. Then there are three
            # child nodes of Health: Diseases, Public health and Obesity. Then when we are trying to fit the Health
            # classifier for the two datum shown above, it will need to bug. Because the classifier is actually fitting
            # a binary classifier to each possible result. In this case, it would be fitting a binary classifier for each
            # of Diseases, Public health and Obesity. As both of these two inputs are True in Diseases. It means that all
            # the data for the binary classifier Diseases are 1, which is not allowed.
            except ValueError:
                node.classifier = base_classifier(self.tree_type)
                temp = np.sum(train_labels, axis=0)
                always_have_label_ind = temp == [train_labels.shape[0] for _ in range(train_labels.shape[1])]
                always_have_labels = node.mlb.inverse_transform(always_have_label_ind.reshape(1, train_labels.shape[1]))
                new_labels = node.mlb.fit_transform([s - set(always_have_labels[0]) for s in node.labels])
                if not new_labels[0].tolist():
                    continue
                    # It means that there is only one child for this node, thus no need to train
                node.ignore_labels = always_have_labels[0]

                node.classifier.fit(train_data, new_labels)

    def predict(self, test_data):
        """
        Predict the labels of test_data.
        :param test_data: Numpy array or sparse matrix.
        :return: [[String]]
        """
        final_labels = []
        for i, line in enumerate(test_data):
            final_label = []
            level_label = ['Top/']
            level_nodes = [self.root]
            while level_nodes:
                temp_nodes = []
                temp_labels = []
                for j, node in enumerate(level_nodes):
                    if not node.classifier:
                        final_label.append(level_label[j])
                        continue
                    # the prediction comes to the bottom of the tree, append it to the result.
                    if node.ignore_labels:
                        for l in node.ignore_labels:
                            temp_nodes.append(node.child[l])
                            temp_labels.append(level_label[j] + l + '/')
                    # add the always have labels(because it cannot be trained) to the bfs list.
                    if not node.ignore_labels or len(node.child.keys()) != len(node.ignore_labels):
                        try:
                            label = node.classifier.predict(line)
                        except NotFittedError:
                            print(node.ignore_labels)
                            print(node.key)
                            print(node.child.keys())
                            final_label.append(level_label[j])
                            continue
                        label = node.mlb.inverse_transform(label)
                        if not label[0]:
                            final_label.append(level_label[j])
                        for l in label[0]:
                            temp_nodes.append(node.child[l])
                            temp_labels.append(level_label[j] + l + '/')
                    # predict the test example and add it to the result.
                level_label = temp_labels
                level_nodes = temp_nodes
            final_labels.append(final_label)
            if i % 1000 == 0:
                print("predict {0} test documents".format(i))

        return final_labels

    def predict_text(self, text, vectorizer):
        """
        This is used when test data is list of strings. Be aware that the vectorizer must be the one that used to
        transform the train data, or it will need to error.
        :param text: List of String
        :return: [[String]]
        """
        test_data = vectorizer.transform(text)
        final_labels = self.predict(test_data)
        return final_labels

    def save_model(self, path):
        """
        Save the model to the given path.
        :param path: String. Path to save the model. Remember to end with a /.
        :return:
        """
        nodes_list = self.traverse()
        nodes_list = [node for node in nodes_list if node]
        meta_info = {}
        for node in nodes_list:
            node_info = {'key': node.key, 'child': list(node.child.keys()), 'level':node.level,
                         'ignore_labels': node.ignore_labels}
            if node.pre_node:
                node_info['pre_node'] = node.pre_node.key
            sub_dir_classifier = path + node.key + '_' + str(node.level) + '_classifier.pkl'
            sub_dir_mlb = path + node.key + '_' + str(node.level) + '_mlb.pkl'
            joblib.dump(node.classifier, sub_dir_classifier)
            joblib.dump(node.mlb, sub_dir_mlb)
            meta_info[node.key + '_' + str(node.level)] = node_info
        with open(path + 'meta_info.pkl', 'wb') as f:
            pickle.dump(meta_info, f, pickle.HIGHEST_PROTOCOL)

def load_model(path):
    """
    load model from the given path
    :param path: String. The path that store all the .pkl files to restore the model.
    :return: root. a Node object that is also the root of the whole tree.
    """
    with open(path + 'meta_info.pkl', 'rb') as f:
        meta_info = pickle.load(f)
    pkl_list = [path + 'Top_0_classifier.pkl']
    root = Node('Top', None, 0)
    nodes = [root]
    while pkl_list:
        pkl = pkl_list.pop(0)
        clf = joblib.load(pkl)
        mlb_path = '_'.join(pkl.split('_')[:-1]) + '_mlb.pkl'
        mlb = joblib.load(mlb_path)
        node = nodes.pop(0)
        node.classifier = clf
        node.mlb = mlb
        node_meta_info = meta_info[node.key + '_' + str(node.level)]
        node.ignore_labels = node_meta_info['ignore_labels']
        if 'pre_node' in node_meta_info:
            node.pre_node = node_meta_info['pre_node']
        children = meta_info[node.key + '_' + str(node.level)]['child']
        for child in children:
            new_node =  Node(child, None, node.level + 1)
            node.child[child] = new_node
            nodes.append(new_node)
            pkl_list.append(path + child + '_' + str(node.level + 1) + '_classifier.pkl')

    return root


class RestoredTree(Tree):
    def __init__(self, model_path):
        """
        Restore the model from model_path and vectorizer_path
        :param model_path:
        :param vectorizer_path:
        """
        self.root = load_model(model_path)







