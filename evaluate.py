import numpy as np


def evaluate(true_labels, pred_labels, level):
    TP, precision_d, recall_d = np.zeros(level), np.zeros(level), np.zeros(level)
    for i in range(len(true_labels)):
        tp, p_d, r_d = evaluate_datum(true_labels[i], pred_labels[i], level)
        TP += tp
        precision_d += p_d
        recall_d += r_d
        if i % 10000 == 0:
            print('processed {0} data'.format(i))
    precision = TP / precision_d
    recall = TP/ recall_d
    F1 = 2 * (precision * recall) / (precision + recall)
    return precision, recall, F1


def evaluate_datum(true_labels, pred_labels, level):
    true_labels = [l.split('/')[:level] for l in true_labels]
    pred_labels = [l.split('/')[:level] for l in pred_labels]
    TP = []
    precision_denominator = []
    recall_denominator = []
    for i in range(1, level + 1):
        true_set = set(['/'.join(l[:i]) for l in true_labels])
        pred_set = set(['/'.join(l[:i]) for l in pred_labels])
        TP.append(len(true_set.intersection(pred_set)))
        precision_denominator.append(len(pred_set))
        recall_denominator.append(len(true_set))
    return np.array(TP), np.array(precision_denominator), np.array(recall_denominator)

if __name__ == '__main__':
    print(evaluate([["Top/News/Sports", "Top/News/Sports/Soccer", "Top/News/Sports/Baseball"]], \
                            [['Top/News/Politics', "Top/News/Sports/Baseball"]], 4))

