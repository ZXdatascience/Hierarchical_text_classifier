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

    with open('test_result_1987_2007.txt', 'r') as f:
        predict = f.read()
        predict = [t for t in predict.split('\n') if t]
        predict_res = []
        for sentence in predict:
            s = sentence.split(' $ ')
            new_s = [t for t in s if t]
            predict_res.append(new_s)
    with open('test_labels_1987_2007.txt', 'r') as f:
        test = f.read()
        test = [t for t in test.split('\n') if t]
        test_res = []
        for sentence in test:
            s = sentence.split(' $ ')
            new_s = [t for t in s if t]
            test_res.append(new_s)

    print(len(predict_res))
    p, r, F1 = evaluate(test_res, predict_res, 6)
    print(p)
    print(r)
    print(F1)
