import incremental_evaluation.utils as IE
import incremental_evaluation.scenario_sets as SS
import incremental_evaluation.visualisation_helper as VH
import models.basic_predictor_interfaces
import models.ensgendel_interface
import incremental_evaluation.data_file_helper as DFH
import os
SS_MNIST012 = "mnist012"
SS_MNIST197 = "mnist197"
SS_MNIST_CN5 = "mnist_cn5"
SS_GAUSS3 = "gauss_3"
RESULTS = os.path.join("results", "incremental_evaluation_run")


def datafile_path(_experiment_name, _scenario_set_name, _trial_tag):
    return os.path.join(RESULTS, "{}_{}_{}".format(_experiment_name, _scenario_set_name, _trial_tag))


def stat_cell_format(stats, iteration):
    return "{:.2f}({:.2f})".format(stats["mean"][iteration], stats["std"][iteration])

def scenario_into_filename(scenario_string):
    return scenario_string.replace(' ', '').replace('[','').replace('],', 'a').replace(']', '').\
        replace('{','T').replace('}','').replace(',','').replace(':','x')



if __name__ == '__main__':

    # Experiment setup
    trial_tags = [0]
    experiment_name = "test1"
    scout_subset = 100
    scenario_set_name = SS_MNIST012
    # scenario_set_name = SS_MNIST_CN5
    mode = []
    # mode += [1]  # show scenario data
    mode += [2]  # run predictor learning on scenarios
    mode += [3]  # evaluate predictors scenarios
    mode += [4]  # write accuracy statistics into table
    mode += [5]  # write accuracy statistics into table

    # list of predictor classes that implement the incremental_evaluation.interfaces.Predictor
    predictor_builders = [
        models.basic_predictor_interfaces.LinearClassifierPredictor,
        models.basic_predictor_interfaces.PerceptronClassifierPredictor,
        models.ensgendel_interface.Ensgendel
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

    # Pre-flight check of the scenario
    if 1 in mode:
        scenarios = scenario_set.get_scenarios()
        train_sam, train_sub = scenario_set.get_training_set()
        test_sam, test_sub = scenario_set.get_test_set()
        for scenario in scenarios:
            VH.show_scenario(scenario, test_sam, test_sub, visualiser)

    # setting up basic directories
    if not os.path.exists("results"):
        os.mkdir("results")
    if not os.path.exists(RESULTS):
        os.mkdir(RESULTS)

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
            fig_path = os.path.join(RESULTS, "{}_{}_{}_accuracy.pdf".format(experiment_name, scenario_set_name, scenario_into_filename(scenario)))
            VH.show_metric_evol(eval_stats_total, scenario, classifier_style, legend_on=(i == 0), fig_path=fig_path)
            print("fig of scenario {} saved into {}".format(scenario, fig_path))
