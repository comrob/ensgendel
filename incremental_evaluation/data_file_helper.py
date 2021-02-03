import incremental_evaluation.utils as IE
import h5py
import numpy as np
import os
import time

TOTAL_ACCURACY = "total_accuracy"
LOCAL_ACCURACY = "local_accuracy"
SUBCLASS_ACCURACY = "subclass_accuracy"


class DataKeys(object):
    """
    The structure of the HDF5 data file
    """
    # file name
    FILE_NAME = "data.hdf5"
    # Group containing evaluation datasets
    DATASETS = "datasets"
    # Group containing evaluation of predictors
    RESULTS = "results"

    class Datasets(object):
        # Dataset containing training samples
        TRAIN_SAMPLES = "train_samples"
        # Dataset containing testing samples
        TEST_SAMPLES = "test_samples"
        # Dataset containing training subclass labels
        TRAIN_SUBCLASSES = "train_subclasses"
        # Dataset containing testing subclass labels
        TEST_SUBCLASSES = "test_subclasses"
        # Attribute with scenario-set string
        SCENARIOS = "scenarios"

    class Results(object):
        # Group with the name of the predictor
        class Predictor(object):
            # Group with the index of the scenario
            class Scenario(object):
                # Attribute containing scenario string
                SCENARIO = "scenario"
                # Attrubute with evaluation start time
                START = "start"
                # Attrubute with evaluation end time
                END = "end"

                # Group with the index of the task
                class Task(object):
                    # Attribute containing task string
                    TASK = "task"
                    # Dataset containing labels over training set
                    TRAIN_PREDICTIONS = "train_predictions"
                    # Dataset containing labels over testing set
                    TEST_PREDICTIONS = "test_predictions"

DK = DataKeys()


def path_join(*keys):
    return '/'.join(keys)


def extract_total_accuracy(data_path, predictor_type, over_testing_set=True, task_accuracy_type=TOTAL_ACCURACY, evaluator=None):
    f = h5py.File(os.path.join(data_path, DK.FILE_NAME), "r")
    _, train_sub, _, test_sub = (
        f[path_join(DK.DATASETS, DK.Datasets.TRAIN_SAMPLES)],
        f[path_join(DK.DATASETS, DK.Datasets.TRAIN_SUBCLASSES)],
        f[path_join(DK.DATASETS, DK.Datasets.TEST_SAMPLES)],
        f[path_join(DK.DATASETS, DK.Datasets.TEST_SUBCLASSES)],
    )
    if over_testing_set:
        eval_sub = test_sub
        eval_predictions = DK.Results.Predictor.Scenario.Task.TEST_PREDICTIONS
    else:
        eval_sub = train_sub
        eval_predictions = DK.Results.Predictor.Scenario.Task.TRAIN_PREDICTIONS
    if task_accuracy_type == TOTAL_ACCURACY:
        task_accuracy_evaluator = IE.evaluate_task_total_accuracy
    elif task_accuracy_type == LOCAL_ACCURACY:
        task_accuracy_evaluator = IE.evaluate_task_accuracy
    else:
        task_accuracy_evaluator = evaluator

    scenarios = eval(f[DK.DATASETS].attrs[DK.Datasets.SCENARIOS])
    grp_results = f[DK.RESULTS]
    accuracies = {}
    grp_pred = grp_results[predictor_type]
    for sc in range(len(scenarios)):
        grp_scenario = grp_pred[str(sc)]
        predictions = [grp_scenario[str(ts)][eval_predictions] for ts in range(len(scenarios[sc]))]
        accuracies[grp_scenario.attrs[DK.Results.Predictor.Scenario.SCENARIO]] = task_accuracy_evaluator(
            scenarios[sc], predictions, eval_sub)
    f.close()
    return accuracies


def scenario_canonic_form(scenario_string):
    scenario =eval(scenario_string)
    for sc in scenario:
        for lab in sc:
            sc[lab] = set(sc[lab])
    return scenario


def scenario_hash_map(scenario_list, global_scenario_list):
    """
    Due to different scenario hashes (strings) there must be additonal map created.
    Probably problem caused by different pythons.
    :param scenario_list:
    :param global_scenario_list:
    :return: map between scenario_list -> global_scenario_list
    """
    global_scenario_dicts = [scenario_canonic_form(scg) for scg in global_scenario_list]
    ret = {}
    for sc in scenario_list:
        sc_dict = scenario_canonic_form(sc)
        for i, scg_dict in enumerate(global_scenario_dicts):
            if sc_dict == scg_dict:
                ret[sc] = global_scenario_list[i]
                break
    return ret


def extract_stats_for_portfolio(portfolio, task_accuracy_type=TOTAL_ACCURACY, over_testing_set=True, evaluator=None):
    """
    Extracts statistics of the evaluator output which is computed over portfolio.
    Args:
        portfolio: dictionary {<predictor_name>: [datafile path, ...], ...}
        task_accuracy_type: TOTAL_ACCURACY or LOCAL_ACCURACY or None
        over_testing_set:
        evaluator: if task_accuracy_type is none: callable evaluator(scenario, predictions, subclasses) -> [float, ...]

    Returns: dictionary {<predictor_name>: {scenario: {min:[],max:[],std:[],mean:[],num:int,metrics:[]}, ...}, ...}

    """
    eval_metrics = {}
    scenarios = []
    # extracting data
    for pred_type in portfolio:
        eval_metrics[pred_type] = {}
        for data_path in portfolio[pred_type]:
            accuracies = extract_total_accuracy(data_path, pred_type,
                                                over_testing_set=over_testing_set,
                                                task_accuracy_type=task_accuracy_type,
                                                evaluator=evaluator
                                                )
            if len(scenarios) == 0:
                scenarios = list(accuracies.keys())
            compatible_scenario_map = scenario_hash_map(accuracies.keys(), scenarios)
            for sc in accuracies:
                if compatible_scenario_map[sc] not in eval_metrics[pred_type]:
                    eval_metrics[pred_type][compatible_scenario_map[sc]] = []
                eval_metrics[pred_type][compatible_scenario_map[sc]].append(accuracies[sc])
    # calculating stats
    eval_stats = {}
    for pred_type in eval_metrics:
        eval_stats[pred_type] = {}
        for sc in eval_metrics[pred_type]:
            metrics = np.asarray(eval_metrics[pred_type][sc])
            eval_stats[pred_type][sc] = {
                "metrics": metrics,
                "mean": np.mean(metrics, axis=0),
                "std": np.std(metrics, axis=0),
                "min": np.min(metrics, axis=0),
                "max": np.max(metrics, axis=0),
                "num": metrics.shape[0]
            }
    return eval_stats


def run_and_save(predictor_classes, scenario_set, experiment_path):
    """
    Runs the predictors in with given scenario set and saves results on experiment path.
    Args:
        predictor_classes: list of classes implementing interfaces.Predictor
        scenario_set: object implementing interfaces.ScenarioSet
        experiment_path:

    Returns:

    """
    # scenario set unpacking
    scenarios = scenario_set.get_scenarios()
    train_sam, train_sub = scenario_set.get_training_set()
    test_sam, test_sub = scenario_set.get_test_set()

    # hdf5 file filling
    f = h5py.File(os.path.join(experiment_path, DK.FILE_NAME), "w")
    # Saving datasets and scenario
    grp_ds = f.create_group(DK.DATASETS)
    grp_ds.attrs[DK.Datasets.SCENARIOS] = str(scenarios)
    grp_ds.create_dataset(DK.Datasets.TRAIN_SAMPLES, data=train_sam)
    grp_ds.create_dataset(DK.Datasets.TEST_SAMPLES, data=test_sam)
    grp_ds.create_dataset(DK.Datasets.TRAIN_SUBCLASSES, data=train_sub, dtype=np.int)
    grp_ds.create_dataset(DK.Datasets.TEST_SUBCLASSES, data=test_sub, dtype=np.int)
    # Saving predictor results
    grp_results = f.create_group(DK.RESULTS)
    for pred_bld in predictor_classes:
        grp = grp_results.create_group(str(pred_bld))
        for sc in range(len(scenarios)):
            grp_sc = grp.create_group(str(sc))
            grp_sc.attrs[DK.Results.Predictor.Scenario.SCENARIO] = str(scenarios[sc])
            grp_sc.attrs[DK.Results.Predictor.Scenario.START] = time.time()
            pred = pred_bld(classes=IE.labels_from_scenario(scenarios[sc]))
            results = IE.run_scenario(scenarios[sc], train_sam, train_sub, test_sam, pred.fit, pred.predict)
            grp_sc.attrs[DK.Results.Predictor.Scenario.END] = time.time()
            for res in range(len(results)):
                result = results[res]
                grp_iter = grp_sc.create_group(str(res))
                grp_iter.attrs[DK.Results.Predictor.Scenario.Task.TASK] = str(scenarios[sc][res])
                grp_iter.create_dataset(DK.Results.Predictor.Scenario.Task.TRAIN_PREDICTIONS,
                                        data=result[0], dtype=np.int)
                grp_iter.create_dataset(DK.Results.Predictor.Scenario.Task.TEST_PREDICTIONS,
                                        data=result[1], dtype=np.int)
    f.close()


def datafile_evaluation(experiment_path, evaluation_method_dictionary, eval_on_testset=True):
    """
    Evaluates predictions in datafile with given evaluation methods.
    Args:
        experiment_path:
        evaluation_method_dictionary: {evaluation_name: evaluation_callable(scenario, predictions, subclasses), ...}

    Returns: dictionary {<predictor_name>: {scenario: {evaluation: [], ...}, ...}, ...}

    """
    f = h5py.File(os.path.join(experiment_path, DK.FILE_NAME), "r")
    if eval_on_testset:
        subclasses = f[path_join(DK.DATASETS, DK.Datasets.TEST_SUBCLASSES)]
        pred_key = DK.Results.Predictor.Scenario.Task.TEST_PREDICTIONS
    else:
        subclasses = f[path_join(DK.DATASETS, DK.Datasets.TRAIN_SUBCLASSES)]
        pred_key = DK.Results.Predictor.Scenario.Task.TRAIN_PREDICTIONS
    scenarios = eval(f[DK.DATASETS].attrs[DK.Datasets.SCENARIOS])
    grp_results = f[DK.RESULTS]
    results = {}
    for predictor_type in grp_results.keys():
        grp_pred = grp_results[predictor_type]
        results[predictor_type] = {}
        for sc in range(len(scenarios)):
            grp_scenario = grp_pred[str(sc)]
            predictions = [grp_scenario[str(ts)][pred_key] for ts in range(len(scenarios[sc]))]
            evals = {}
            for eval_meth in evaluation_method_dictionary:
                evals[eval_meth] = evaluation_method_dictionary[eval_meth](scenarios[sc], predictions, subclasses)
            results[predictor_type][grp_scenario.attrs[DK.Results.Predictor.Scenario.SCENARIO]] = evals
    f.close()
    return results