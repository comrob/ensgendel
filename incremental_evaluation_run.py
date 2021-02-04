import incremental_evaluation.utils as IE
import incremental_evaluation.scenario_sets as SS
import incremental_evaluation.visualisation_helper as VH
import models.basic_predictor_interfaces
import models.ensgendel_interface
import incremental_evaluation.data_file_helper as DFH
import os
import argparse

SS_MNIST012 = "mnist012"
SS_MNIST197 = "mnist197"
SS_MNIST_CN5 = "mnist_cn5"
SS_GAUSS3 = "gauss_3"
RESULTS = os.path.join("results", "incremental_evaluation_run")


def datafile_path(_experiment_name, _scenario_set_name, _trial_tag):
    return os.path.join(RESULTS, "{}_{}_{}".format(_experiment_name, _scenario_set_name, _trial_tag))


def stat_cell_format(stats, iteration):
    return "{:.2f}({:.2f})".format(stats["mean"][iteration], stats["std"][iteration])

parser = argparse.ArgumentParser(description="EnsGenDel algorithm & Incremental evaluation framework")
parser.add_argument('experiment_name', help="Experiment name which will be in file prefix.")
parser.add_argument('scenario_name', help="Select the scenario. One of the following: " + str([
    SS_MNIST012, SS_MNIST197, SS_MNIST_CN5, SS_GAUSS3]) + "The scenario name is appended after experiment_name.")
parser.add_argument('modes',
                    help="Series of numbers activating five modes of this application:"
                         "1-scenario preview, 2-predictor training, 3-debug evaluation,"
                         " 4-generate csv table with evaluation stats, 5-generate accuracy plots"
                         ";e.g., '24' trains the predictors and thes generates csv table with results.")
parser.add_argument('--trials', type=int, default=1, help="Number of independend runs. The trial number is appended "
                                                          "in the postfix of the file.")
parser.add_argument('--trials_from', type=int, default=0, help="Index of the first trial.")
parser.add_argument('--scout_number', type=int, default=-1, help="Cropping the training set. Speeding up the training "
                                                                 "at the cost of less accuracy.")
parser.add_argument("--debug", default=False, type=bool, help="Runs only light weight models. True/False")

if __name__ == '__main__':
    args = parser.parse_args()

    # Experiment setup
    trial_tags = [i for i in range(args.trials_from, args.trials_from + args.trials)]
    experiment_name = args.experiment_name
    scout_subset = args.scout_number if args.scout_number > 0 else None
    scenario_set_name = args.scenario_name
    mode = list(map(int, args.modes))
    # mode += [1]  # show scenario data
    # mode += [2]  # run predictor learning on scenarios
    # mode += [3]  # evaluate predictors scenarios
    # mode += [4]  # write accuracy statistics into table
    # mode += [5]  # write accuracy statistics into table

    # list of predictor classes that implement the incremental_evaluation.interfaces.Predictor
    predictor_builders = [
        models.basic_predictor_interfaces.LinearClassifierPredictor,
        models.basic_predictor_interfaces.PerceptronClassifierPredictor,
        ]
    if not args.debug:
        predictor_builders += [
            models.ensgendel_interface.Ensgendel,
            models.ensgendel_interface.Ensgen,
            models.ensgendel_interface.Ens,
        ]

    # scenario sets implementing the incremental_evaluation.interfaces.ScenarioSet
    if scenario_set_name == SS_MNIST012:
        scenario_set = SS.MnistMinimalScenarios(digits_tripplet=(0, 1, 2), debug_set=False, scout_subset=scout_subset)
        visualiser = VH.mnist_visualiser
    elif scenario_set_name == SS_MNIST197:
        scenario_set = SS.MnistMinimalScenarios(digits_tripplet=(1, 9, 7), debug_set=False, scout_subset=scout_subset)
        visualiser = VH.mnist_visualiser
    elif scenario_set_name == SS_MNIST_CN5:
        scenario_set = SS.MnistConvergentFiveScenarios(scout_subset=scout_subset)
        visualiser = VH.mnist_visualiser
    elif scenario_set_name == SS_GAUSS3:
        scenario_set = SS.Gauss3DMinimalScenarios(train_size=scout_subset)
        visualiser = VH.gauss3d_visualiser
    else:
        raise NotImplementedError(scenario_set_name)

    # setting up basic directories
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(RESULTS):
        os.mkdir(RESULTS)

    # Pre-flight check of the scenario
    if 1 in mode:
        scenarios = scenario_set.get_scenarios()
        train_sam, train_sub = scenario_set.get_training_set()
        test_sam, test_sub = scenario_set.get_test_set()
        for scenario in scenarios:
            folder_name = "preview_{}".format(VH.scenario_into_filename(str(scenario)))
            folder_path = os.path.join(RESULTS, folder_name)
            if not os.path.exists(folder_path):
                os.mkdir(folder_path)
            VH.show_scenario(scenario, test_sam, test_sub, visualiser, save_into=folder_path)

    # Cycle of experiment runs
    for trial_tag in trial_tags:
        experiment_path = datafile_path(experiment_name, scenario_set_name, trial_tag)
        if not os.path.exists(experiment_path):
            os.mkdir(experiment_path)

        if 2 in mode:
            DFH.run_and_save(predictor_builders, scenario_set, experiment_path)

        if 3 in mode:
            evals = DFH.datafile_evaluation(experiment_path, {
                DFH.TOTAL_ACCURACY: IE.evaluate_task_total_accuracy,
                DFH.LOCAL_ACCURACY: IE.evaluate_task_accuracy,
                DFH.SUBCLASS_ACCURACY: IE.evaluate_subclass_accuracy,
            })
            print(evals)

    # Stats evaluation
    files = [datafile_path(experiment_name, scenario_set_name, trial_tag) for trial_tag in trial_tags]
    portfolio = dict([(str(clazz), files) for clazz in predictor_builders])

    if 4 in mode:
        eval_stats_total = DFH.extract_stats_for_portfolio(portfolio, over_testing_set=True,
                                                           task_accuracy_type=DFH.TOTAL_ACCURACY)
        table = VH.stats_into_text_table(eval_stats_total, stat_cell_format, cell_join=';', row_join='\n')
        print(table)
        table_path = os.path.join(RESULTS, "{}_{}_total_accuracy.csv".format(experiment_name, scenario_set_name))
        with open(table_path, "w") as fil:
            fil.write(table)
        print("Saved stats of total accuracy into {}".format(table_path))

    if 5 in mode:
        figure_styles = [
            [("color", "r"), ("marker", "o")],
            [("color", "g"), ("marker", "^")],
            [("color", "b"), ("marker", "x")],
            [("color", "c"), ("marker", "s")],
            [("color", "m"), ("marker", "d")],
            [("color", "y"), ("marker", "+")],
            [("color", "k"), ("marker", "*")],
        ]
        classifier_style = dict(
            [(str(clazz), dict([("label", clazz.__name__)] + figure_styles[i % len(figure_styles)]))
             for i, clazz in enumerate(predictor_builders)]
        )

        eval_stats_total = DFH.extract_stats_for_portfolio(portfolio, over_testing_set=True,
                                                           task_accuracy_type=DFH.TOTAL_ACCURACY)
        scenarios = list(eval_stats_total[list(eval_stats_total.keys())[0]].keys())
        print(scenarios)
        for i, scenario in enumerate(scenarios):
            # picking subclass for tracking
            scenario_obj = eval(scenario)
            tracked_label = list(scenario_obj[0].keys())[0]
            tracked_subclass = scenario_obj[0][tracked_label][-1]

            def tracked_evaluation(_scen, _pred, _subs): # lambda for tracking
                return IE.evaluate_selected_subclass_accuracy(_scen, _pred, _subs, tracked_subclass, tracked_label)
            eval_stats_tracked = DFH.extract_stats_for_portfolio(
                portfolio, over_testing_set=True, task_accuracy_type=None, evaluator=tracked_evaluation)
            # visualisaiton
            fig_path = os.path.join(RESULTS, "{}_{}_{}_accuracy.pdf".format(experiment_name, scenario_set_name,
                                                                            VH.scenario_into_filename(scenario)))
            VH.show_metric_evol(eval_stats_total, scenario, classifier_style,
                                fig_path=fig_path, selected_eval_stats=eval_stats_tracked, title=scenario)
            print("fig of scenario {} saved into {}".format(scenario, fig_path))
