from PyQt5.QtCore import Qt, QCoreApplication

import itertools
import pandas as pd
import numpy as np
from tqdm.auto import tqdm
import networkx as nx
from collections import Counter

from algorithms.chameleon.chameleon2 import knn_graph_sym, merge_best2, prepro_edge, connected_components, \
    tree_height, first_jump_cutoff, find_nearest_height
from algorithms.chameleon.graphtools import pre_part_graph, get_cluster, connecting_edges
from algorithms.chameleon.visualization import plot2d_data
from algorithms.chameleon.chameleon import rebuild_labels

from GUI_classes.utils_gui import choose_dataset, pause_execution

from GUI_classes.generic_gui import StartingGui, GraphWindow

# TODO: take care of autoextract
# TODO: fix everything on mac and try on windows

class CHAMELEON2_class(StartingGui):
    def __init__(self):
        super(CHAMELEON2_class, self).__init__(name="CHAMELEON2", twinx=False, first_plot=True, second_plot=False,
                                               function=self.start_CHAMELEON2, extract=False, stretch_plot=False)

    def start_CHAMELEON2(self):
        self.ax1.cla()
        self.log.clear()
        self.log.appendPlainText("{} LOG".format(self.name))

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.n_clust = int(self.line_edit_n_clust.text())
        self.knn_cham = int(self.line_edit_knn_cham.text())
        self.init_clust_cham = int(self.line_edit_init_clust_cham.text())
        self.alpha_cham = float(self.line_edit_alpha_cham.text())
        self.beta_cham = float(self.line_edit_beta_cham.text())
        self.m_fact = int(self.line_edit_m_fact.text())
        self.n_points = int(self.line_edit_np.text())

        self.X = choose_dataset(self.combobox.currentText(), self.n_points)

        self.button_run.setEnabled(False)
        self.checkbox_saveimg.setEnabled(False)
        self.button_delete_pics.setEnabled(False)
        self.button_examples_graph.setEnabled(False)
        self.slider.setEnabled(False)

        if self.first_run_occurred is True:
            self.ind_run += 1
            self.ind_extr_fig = 0
            if self.save_plots is True:
                self.checkBoxChangedAction(self.checkbox_saveimg.checkState())
        else:
            if Qt.Checked == self.checkbox_saveimg.checkState():
                self.first_run_occurred = True
                self.checkBoxChangedAction(self.checkbox_saveimg.checkState())

        self.checkbox_gif.setEnabled(False)

        res, h = self.cluster2_gui(pd.DataFrame(self.X), k=self.n_clust, knn=self.knn_cham, m=self.init_clust_cham,
                                   alpha=self.alpha_cham, beta=self.beta_cham, m_fact=self.m_fact, plot=True,
                                   auto_extract=False)

        # plot2d_data(res)

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        self.button_run.setEnabled(True)
        self.button_examples_graph.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        if self.checkbox_saveimg.isChecked() is True:
            self.checkbox_gif.setEnabled(True)
        self.button_delete_pics.setEnabled(True)
        self.slider.setEnabled(True)

    def cluster2_gui(self, df, k=None, knn=None, m=30, alpha=2.0, beta=1, m_fact=1e3,
                     verbose=True, verbose2=True, plot=True, auto_extract=False):
        if knn is None:
            knn = int(round(2 * np.log(len(df))))

        if k is None:
            k = 1

        self.log.appendPlainText("Building kNN graph (k = {})...".format(knn))
        graph_knn = knn_graph_sym(df, knn, verbose)

        self.plot2d_graph_gui(graph_knn, print_clust=True)

        graph_pp = pre_part_graph(graph_knn, m, df, verbose, plotting=plot)

        self.log.appendPlainText("flood fill...")

        graph_ff, increased_m = self.flood_fill_gui(graph_pp, graph_knn, df)

        m = increased_m

        self.log.appendPlainText("new m: {}".format(m))

        self.plot2d_graph_gui(graph_ff, print_clust=True)

        dendr_height = {}
        iterm = tqdm(enumerate(range(m - k)), total=m - k) if verbose else enumerate(range(m - k))

        for i, _ in iterm:

            df, ms, ci = merge_best2(graph_ff, df, alpha, beta, m_fact, k, False, verbose2)

            if ms == 0:
                break

            dendr_height[m - (i + 1)] = ms

            if plot:
                plot2d_data(df, ci)

        self.log.appendPlainText("dendr_height: {}".format(dendr_height))
        res = rebuild_labels(df)

        if auto_extract is True:
            self.extract_optimal_n_clust_gui(dendr_height, m)

        return res, dendr_height

    def flood_fill_gui(self, graph, knn_gr, df):

        cl_dict = {list(graph.node)[i]: graph.node[i]["cluster"] for i in range(len(graph))}
        new_cl_ind = max(cl_dict.values()) + 1
        dic_edge = prepro_edge(knn_gr)

        for num in range(max(cl_dict.values()) + 1):
            points = [i for i in list(cl_dict.keys()) if list(cl_dict.values())[i] == num]
            restr_dict = {list(dic_edge.keys())[i]: dic_edge[i] for i in points}
            r_dict = {}

            for i in list(restr_dict.keys()):
                r_dict[i] = [i for i in restr_dict[i] if i in points]

            cc_list = list(connected_components(r_dict))
            self.log.appendPlainText("num_cluster: {0}, len: {1}".format(num, len(cc_list)))
            if len(cc_list) == 1:
                continue
            else:
                # skip the first
                for component in cc_list[1:]:
                    self.log.appendPlainText("new index for the component: {}".format(new_cl_ind))
                    for el in component:
                        cl_dict[el] = new_cl_ind
                    new_cl_ind += 1

        df["cluster"] = list(cl_dict.values())

        for i in range(len(graph)):
            graph.node[i]["cluster"] = cl_dict[i]

        increased_m = max(cl_dict.values()) + 1

        return graph, increased_m

    def extract_optimal_n_clust_gui(self, h, m, f=1000, eta=2):
        th = tree_height(h, m)

        if len(th) <= 3:
            self.log.appendPlainText("insufficient merging steps to perform auto_extract; "
                                     "decrease k and/or increase m")

            return

        fjc = first_jump_cutoff(th, f, eta, m)

        opt_n_clust = find_nearest_height(th, fjc)

        self.log.appendPlainText("Optimal number of clusters: {}".format(opt_n_clust))

    def plot2d_graph_gui(self, graph, print_clust=True):
        pos = nx.get_node_attributes(graph, 'pos')
        colors = {0: "seagreen", 1: 'beige', 2: 'yellow', 3: 'grey', 4: 'pink', 5: 'turquoise',
                  6: 'orange', 7: 'purple', 8: 'yellowgreen', 9: 'olive', 10: 'brown',
                  11: 'tan', 12: 'plum', 13: 'rosybrown', 14: 'lightblue', 15: "khaki",
                  16: "gainsboro", 17: "peachpuff", 18: "lime", 19: "peru",
                  20: "dodgerblue", 21: "teal", 22: "royalblue", 23: "tomato",
                  24: "bisque", 25: "palegreen"}

        el = nx.get_node_attributes(graph, 'cluster').values()
        cmc = Counter(el).most_common()
        c = [colors[i % len(colors)] for i in el]

        if print_clust is True:
            self.log.appendPlainText("clusters: {}".format(cmc))

        if len(el) != 0:  # is set
            # print(pos)
            nx.draw(graph, pos, node_color=c, node_size=60, edgecolors="black", ax=self.ax1)
        else:
            nx.draw(graph, pos, node_size=60, edgecolors="black", ax=self.ax1)
