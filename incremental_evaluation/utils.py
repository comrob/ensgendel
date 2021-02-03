import numpy as np

"""
Evaluation of incremental learning algorithms.
- samples
List of N D-dimensional features.

- subclasses
List of N markers for each sample which describes to which subclass the sample belongs.
Subclass can be understood as a hidden cluster, which is known only to evaluator.
The subclass is used for construction of incremental scenarios.

- labels
Set of integers which are assigned to samples according to the scenarios. The learning algorithm is trained to
predict the label conditioned on the input sample.

- task map
A map between labels and subclasses. 

- scenario
Sequence of task maps

"""
NULL_LABEL = -1

# Minimal scenarios
SCENARIO_ADDITION = [{0: [0]}, {1: [2]}]
SCENARIO_EXPANSION = [{0: [0]}, {0: [1], 1: [2]}]
SCENARIO_INCLUSION = [{0: [0], 1: [1]}, {0: [1], 1: [2]}]
SCENARIO_SEPARATION = [{0: [0, 1]}, {1: [1, 2]}]

# Monotone scenarios


def subclasses_from_scenario(scenario):
    subclasses = []
    for task in scenario:
        for label in task.keys():
            subclasses += task[label]
    return list(set(subclasses))


def labels_from_scenario(scenario):
    labels = []
    for task in scenario:
        labels += task.keys()
    return list(set(labels))


def get_subclass_labels(subclasses, task_map, null_label=NULL_LABEL):
    """
    Maps the subclasses array to label array. However, since the map is not surjective, the subclasses without label
    are mapped to null_label. Labels are integers.
    :param subclasses: N subclass markers for each sample, [subclass, ...]
    :param task_map: maps label to subclass {(int) label: [subclass, ...], ...}
    :return: [(int) label, ...]
    """
    labels = np.ones((subclasses.shape[0],), dtype=np.int) * null_label
    for l in task_map.keys():
        for sub in task_map[l]:
            labels[np.equal(subclasses, sub)] = l
    return labels


def get_perfect_task_map(scenario, iteration, null_label=NULL_LABEL):
    """
    Computes a perfect ground truth mapping from labels to subclasses (task_map type).
    The algorithm tracks the history of subclass transactions between labels until the given iteration.
    The final state is then converted into the task_map which can be used for evaluation.
    :param scenario: list of task maps
    :param iteration:
    :param null_label:
    :return: task_map type
    """
    subclasses = subclasses_from_scenario(scenario)
    sub_to_lab = dict(zip(subclasses, [null_label] * len(subclasses)))
    # recording history
    for i in range(iteration + 1):
        for label in scenario[i]:
            for sub in scenario[i][label]:
                sub_to_lab[sub] = label
    # inverting the map into the task_map type
    task_map = {}
    for sub in sub_to_lab.keys():
        if sub_to_lab[sub] not in task_map:
            task_map[sub_to_lab[sub]] = []
        task_map[sub_to_lab[sub]].append(sub)
    return task_map


def get_train_set_labels(scenario, iteration, subclasses, null_label=NULL_LABEL):
    return get_subclass_labels(subclasses, scenario[iteration], null_label=null_label)


def get_test_set_labels(scenario, iteration, subclasses, null_label=NULL_LABEL):
    perfect_task = get_perfect_task_map(scenario, iteration, null_label=null_label)
    return get_subclass_labels(subclasses, perfect_task, null_label=null_label)


def run_scenario(scenario, training_samples, training_subclasses, testing_samples,
                 fit, predict, null_label=NULL_LABEL):
    """
    Runs scenario with given train and predict functions and returns the predictions for each scenario iteration.
    :param null_label:
    :param testing_samples: list of features for testing
    :param training_subclasses: list of subclass markers of training_samples
    :param training_samples: list of features for training
    :param scenario: sequence of task maps, [{label: [subclass, ...], ...}, ...]
    :param fit: training function, train(samples, labels) <- changes the predictor
    :param predict: prediction function, predict(samples) -> labels
    :return: list of predicted labels of training and testing samples for each scenario iteration,
    [([label of training sample, ...], [label of testing sample, ...]), ...]
    """
    results = []
    for iteration in range(len(scenario)):
        # training
        training_labels = get_train_set_labels(scenario, iteration, training_subclasses, null_label=null_label)
        active_rows = np.where(training_labels != null_label)[0]
        np.random.shuffle(active_rows)
        fit(training_samples[active_rows], training_labels[active_rows])
        # getting predictions
        trainset_predictions = predict(training_samples)
        testset_predictions = predict(testing_samples)
        results.append((trainset_predictions, testset_predictions))
    return results


def _accuracy(true_labels, predicted_labels):
    is_same = np.equal(true_labels, predicted_labels)
    hits = np.sum(is_same)
    return float(hits)/is_same.shape[0]


def evaluate_task_accuracy(scenario, prediction_results, subclasses, null_label=NULL_LABEL):
    """
    Evaluates the accuracy of the predictor for each task, where we consider the best possible performance as
    the ground truth. The best possible is the perfect labeling for current iteration.
    :param scenario:
    :param prediction_results:
    :param subclasses:
    :param null_label:
    :return:
    """
    accuracies = []
    for iteration in range(len(scenario)):
        true_labels = get_test_set_labels(scenario, iteration, subclasses, null_label=null_label)
        active_rows = true_labels != null_label
        accuracies.append(_accuracy(true_labels[active_rows], prediction_results[iteration][active_rows]))
    return accuracies


def evaluate_task_total_accuracy(scenario, prediction_results, subclasses, null_label=NULL_LABEL):
    """
    Evaluates the accuracy of the predictor for each task, where the ground truth is the best final labeling
    as the ground truth. The best possible is the perfect labeling for last iteration.
    :param scenario:
    :param prediction_results:
    :param subclasses:
    :param null_label:
    :return:
    """
    accuracies = []
    true_labels = get_test_set_labels(scenario, len(scenario) - 1, subclasses, null_label=null_label)
    active_rows = true_labels != null_label
    for iteration in range(len(scenario)):
        accuracies.append(_accuracy(true_labels[active_rows], prediction_results[iteration][active_rows]))
    return accuracies


def evaluate_subclass_accuracy(scenario, prediction_results, subclasses, null_label=NULL_LABEL):
    accuracies = []
    for iteration in range(len(scenario)):
        perfect_task = get_perfect_task_map(scenario, iteration)
        task = {}
        for lab in perfect_task:
            if lab == null_label:
                continue
            task[lab] = {}
            for sub in perfect_task[lab]:
                sub_preds = prediction_results[iteration][np.equal(subclasses,sub)]
                task[lab][sub] = np.average(np.equal(sub_preds, lab))

        accuracies.append(task)
    return accuracies


def evaluate_selected_subclass_accuracy(scenario, prediction_results, subclasses, sel_subclass, sel_label, null_label=NULL_LABEL):
    sub_accs = evaluate_subclass_accuracy(scenario, prediction_results, subclasses, null_label=null_label)
    accs = [0]*len(scenario)
    for iteration in range(len(scenario)):
        if sel_label in sub_accs[iteration]:
            if sel_subclass in sub_accs[iteration][sel_label]:
                accs[iteration] = sub_accs[iteration][sel_label][sel_subclass]
    return accs
