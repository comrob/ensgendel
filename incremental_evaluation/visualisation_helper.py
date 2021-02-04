import numpy as np
from incremental_evaluation import utils as IE
import matplotlib.pyplot as plt
from incremental_evaluation.scenario_sets import Gauss3DMinimalScenarios
import os

def scenario_into_filename(scenario_string):
    """
    Since brackets aren't allowed in filenames, scenario is encoded into something more saveable.
    """
    return scenario_string.replace(' ', '').replace('[','').replace('],', 'a').replace(']', '').\
        replace('{','T').replace('}','').replace(',','').replace(':','x')

def mnist_visualiser(figure, active_samples, active_labels, title, sample_num=10):
    label_set = np.unique(active_labels)
    f = figure.subplots(len(label_set), sample_num)
    figure.suptitle(title)
    for i in range(len(label_set)):
        sel_samples = active_samples[active_labels == label_set[i], :]
        for j in range(sample_num):
            if len(label_set) > 1:
                fig = f[i][j]
            else:
                fig = f[j]
            pixels = sel_samples[j, :].reshape((28, 28))
            fig.imshow(pixels, cmap='gray')
            if j == 0:
                fig.set_ylabel("desc: {}".format(label_set[i]))


def gauss3d_visualiser(figure, active_samples, active_labels, title, sample_num=100):
    colors = ["c", "m", "y"]
    A = np.asarray([[-1, -1], [0, -1], [-1, 0]]) * 0.5
    label_set = np.unique(active_labels)
    f = figure.subplots(1, 1)
    figure.suptitle(title)
    trns_samples = active_samples.dot(A)
    for i in range(len(label_set)):
        lab = label_set[i]
        sel_samp = trns_samples[np.equal(active_labels, lab), :]
        f.scatter(sel_samp[:sample_num, 0], sel_samp[:sample_num, 1], color=colors[lab], label="lab: {}".format(lab))

    r_vec = np.asarray(Gauss3DMinimalScenarios.R_MEAN).dot(A)
    g_vec = np.asarray(Gauss3DMinimalScenarios.G_MEAN).dot(A)
    b_vec = np.asarray(Gauss3DMinimalScenarios.B_MEAN).dot(A)

    f.plot(r_vec[0], r_vec[1], "ro")
    f.plot(g_vec[0], g_vec[1], "go")
    f.plot(b_vec[0], b_vec[1], "bo")
    f.legend()


def show_scenario(scenario, samples, subclasses, visualiser, show_plot=False, save_into=None):
    fig_cou = 0
    visualiser(plt.figure(fig_cou), samples, subclasses, "Subclasses of scenario {}".format(scenario))
    if save_into is not None:
        name = "preview_all_subclasses.pdf"
        plt.savefig(os.path.join(save_into, name), bbox_inches='tight')

    for i in range(len(scenario)):
        labels = IE.get_train_set_labels(scenario, i, subclasses)
        active_rows = np.where(labels != IE.NULL_LABEL)[0]
        np.random.shuffle(active_rows)
        active_samples = samples[active_rows, :]
        active_labels = labels[active_rows]

        fig_cou += 1
        fig = plt.figure(fig_cou)
        visualiser(fig, active_samples, active_labels, "train task [{}]: {}".format(i, scenario[i]))
        if save_into is not None:
            name = "preview{}_{}_train.pdf".format(i, scenario_into_filename(str(scenario[i])))
            plt.savefig(os.path.join(save_into, name), bbox_inches='tight')

    for i in range(len(scenario)):
        labels = IE.get_test_set_labels(scenario, i, subclasses)
        perfect_task = IE.get_perfect_task_map(scenario, i)
        active_rows = np.where(labels != IE.NULL_LABEL)[0]
        np.random.shuffle(active_rows)
        active_samples = samples[active_rows, :]
        active_labels = labels[active_rows]

        fig_cou += 1
        fig = plt.figure(fig_cou)
        visualiser(fig, active_samples, active_labels, "test task [{}] {}, perfect: {}".format(i, scenario[i], perfect_task))
        if save_into is not None:
            name = "preview{}_{}_test.pdf".format(i, scenario_into_filename(str(scenario[i])))
            plt.savefig(os.path.join(save_into, name), bbox_inches='tight')
    if show_plot:
        plt.show()

def stats_into_text_table(eval_stats, stat_cell_fromatter, cell_join="& ", row_join="\\\\\n",
                          nice_predictor_names=None, force_scenario_order=None, nice_scenario_names=None):
    scenarios = []
    scenario_lengths = []
    first_pred = True
    rows = []
    for pred_type in eval_stats:
        if nice_predictor_names is None:
            row = [pred_type]
        else:
            row = [nice_predictor_names[pred_type]]
        if first_pred:
            if force_scenario_order is None:
                scenarios = list(eval_stats[pred_type].keys())
            else:
                scenarios = force_scenario_order
                assert set(eval_stats[pred_type].keys()) == set(force_scenario_order), "forced scenarios must match with contained ones"
        for i, sc in enumerate(scenarios):
            stats = eval_stats[pred_type][sc]
            if first_pred:
                scenario_lengths.append(stats["metrics"].shape[1])
            for task in range(scenario_lengths[i]):
                row.append(stat_cell_fromatter(stats, task))
        first_pred = False
        rows.append(row)
    header = ["predictors"]
    for i, sc in enumerate(scenarios):
        if nice_scenario_names is None:
            header += [str(sc)] + [""]*(scenario_lengths[i]-1)
        else:
            header += [nice_scenario_names[sc]] + [""] * (scenario_lengths[i] - 1)

    table = [header] + rows
    return row_join.join(map(cell_join.join, table))


def show_metric_evol(eval_stats, scenario, classifier_style, selected_eval_stats=None, legend_on=True, fig_path=None,
                     title=None, show_plot=False):
    plt.rcParams["figure.figsize"] = (6, 3)
    fig = plt.figure()
    if title is not None:
        fig.suptitle(title)
    fig.subplots_adjust(
        left=0.15, bottom=0.20, right=.98, top=0.83,
        wspace=0.12, hspace=0.05)
    figr = fig.subplots(1, 1)
    for pred_type in eval_stats:
        cls_name = pred_type
        results = eval_stats[pred_type][scenario]

        tasks_num = results["mean"].shape[0]
        figr.plot(results["mean"],
                  marker=classifier_style[cls_name]["marker"],
                  color=classifier_style[cls_name]["color"],
                  label=classifier_style[cls_name]["label"],
                  linestyle="-")
        figr.fill_between([i for i in range(results["min"].shape[0])],results["min"], results["max"], color=classifier_style[cls_name]["color"], alpha=0.2)
        if selected_eval_stats is not None:
            detail_results = selected_eval_stats[pred_type][scenario]
            figr.plot(detail_results["mean"],
                      marker=classifier_style[cls_name]["marker"],
                      color=classifier_style[cls_name]["color"],
                      linestyle="--")
    figr.set_xlabel(r"task")
    figr.set_ylabel(r"accuracy")
    figr.set_xticks([t for t in range(tasks_num)])
    figr.set_xticklabels([r"T_{}".format(t + 1) for t in range(tasks_num)])
    figr.set_yticks([i*0.25 for i in range(5)])
    if legend_on:
        figr.legend()
    if fig_path is not None:
        plt.savefig(fig_path, bbox_inches='tight')

    ###################################
    if show_plot:
        plt.show()