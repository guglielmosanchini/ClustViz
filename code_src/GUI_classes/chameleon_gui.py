import metis
from PyQt5.QtCore import Qt, QCoreApplication

import itertools
import pandas as pd
import numpy as np
import networkx as nx
from collections import Counter

from algorithms.chameleon.graphtools import (
    knn_graph,
    get_cluster,
    connecting_edges,
)
from algorithms.chameleon.chameleon import rebuild_labels, merge_score

from GUI_classes.utils_gui import choose_dataset, pause_execution

from GUI_classes.generic_gui import StartingGui


# TODO: fix everything on mac and try on windows


class CHAMELEON_class(StartingGui):
    def __init__(self):
        super(CHAMELEON_class, self).__init__(
            name="CHAMELEON",
            twinx=False,
            first_plot=False,
            second_plot=False,
            function=self.start_CHAMELEON,
            extract=False,
            stretch_plot=False,
        )

        self.SetWindowsCHAMELEON()

    def start_CHAMELEON(self):

        self.ind_fig = 0
        self.SetWindowsCHAMELEON()
        self.log.clear()
        self.log.appendPlainText("{} LOG".format(self.name))
        QCoreApplication.processEvents()

        self.verify_input_parameters()

        if self.param_check is False:
            return

        self.n_clust = int(self.line_edit_n_clust.text())
        self.knn_cham = int(self.line_edit_knn_cham.text())
        self.init_clust_cham = int(self.line_edit_init_clust_cham.text())
        self.alpha_cham = float(self.line_edit_alpha_cham.text())
        self.n_points = int(self.line_edit_np.text())

        self.X = choose_dataset(self.combobox.currentText(), self.n_points)

        self.button_run.setEnabled(False)
        self.checkbox_saveimg.setEnabled(False)
        self.button_delete_pics.setEnabled(False)
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

        res, h = self.cluster_gui(
            pd.DataFrame(self.X),
            k=self.n_clust,
            knn=self.knn_cham,
            m=self.init_clust_cham,
            alpha=self.alpha_cham,
            save_plots=self.save_plots,
        )

        self.plot2d_data_gui(
            res,
            canvas=self.canvas_down,
            ax=self.ax,
            save_plots=self.save_plots,
            ind_fig=self.ind_fig,
        )

        if (self.make_gif is True) and (self.save_plots is True):
            self.generate_GIF()

        self.button_run.setEnabled(True)
        self.checkbox_saveimg.setEnabled(True)
        if self.checkbox_saveimg.isChecked() is True:
            self.checkbox_gif.setEnabled(True)
        self.button_delete_pics.setEnabled(True)
        self.slider.setEnabled(True)

    def cluster_gui(self, df, k, knn=10, m=30, alpha=2.0, save_plots=None):
        if k is None:
            k = 1

        self.log.appendPlainText("Building kNN graph (k={})\n...".format(knn))
        self.log.appendPlainText("")
        graph = knn_graph(df, knn, False)

        self.plot2d_graph_gui(
            graph=graph,
            canvas=self.canvas_up,
            ax=self.ax1,
            save_plots=save_plots,
            ind_fig=self.ind_fig,
            print_clust=False,
        )

        graph = self.pre_part_graph_gui(
            graph=graph,
            canvas=self.canvas_up,
            ax=self.ax1,
            k=m,
            df=df,
            plotting=True,
        )

        # to account for cases where initial_clust is too big or k is already reached before the merging phase
        cl_dict = {
            list(graph.node)[i]: graph.node[i]["cluster"]
            for i in range(len(graph))
        }
        m = len(Counter(cl_dict.values()))
        self.log.appendPlainText("")
        self.log.appendPlainText("actual init_clust: {}".format(m))
        self.log.appendPlainText("")

        dendr_height = {}
        iterm = enumerate(range(m - k))

        for i, _ in iterm:

            df, ms, ci = self.merge_best_gui(
                graph, df, alpha, k, verbose=False, verbose2=True
            )

            if ms == 0:
                break

            dendr_height[m - (i + 1)] = ms

            self.plot2d_data_gui(
                df=df,
                col_i=ci,
                canvas=self.canvas_down,
                ax=self.ax,
                save_plots=save_plots,
                ind_fig=self.ind_fig,
            )
            self.ind_fig += 1

        res = rebuild_labels(df)

        return res, dendr_height

    def merge_best_gui(self, graph, df, a, k, verbose=True, verbose2=True):
        clusters = np.unique(df["cluster"])
        max_score = 0
        ci, cj = -1, -1
        if len(clusters) <= k:
            return False

        for combination in itertools.combinations(clusters, 2):
            i, j = combination
            if i != j:
                if verbose:
                    self.log.appendPlainText(
                        "Checking c{} and c{}".format(i, j)
                    )
                gi = get_cluster(graph, [i])
                gj = get_cluster(graph, [j])
                edges = connecting_edges((gi, gj), graph)
                if not edges:
                    continue
                ms = merge_score(graph, gi, gj, a)
                if verbose:
                    self.log.appendPlainText(
                        "Merge score: {}".format(round(ms, 4))
                    )
                if ms > max_score:
                    if verbose:
                        self.log.appendPlainText(
                            "Better than: {}".format(round(max_score, 4))
                        )
                    max_score = ms
                    ci, cj = i, j

        if max_score > 0:

            if verbose2:
                self.log.appendPlainText("Merging c{} and c{}".format(ci, cj))
                self.log.appendPlainText(
                    "score: {}".format(round(max_score, 4))
                )
                self.log.appendPlainText("")

            df.loc[df["cluster"] == cj, "cluster"] = ci

            for i, p in enumerate(graph.nodes()):
                if graph.node[p]["cluster"] == cj:
                    graph.node[p]["cluster"] = ci
        else:
            self.log.appendPlainText("No Merging")
            self.log.appendPlainText("score: {}".format(round(max_score, 4)))
            self.log.appendPlainText("early stopping")

        return df, max_score, ci

    def pre_part_graph_gui(
        self, graph, k, canvas, ax, df=None, plotting=False
    ):

        self.ind_fig = 1

        self.log.appendPlainText("Begin clustering...")
        clusters = 0
        for i, p in enumerate(graph.nodes()):
            graph.node[p]["cluster"] = 0
        cnts = {0: len(graph.nodes())}

        while clusters < k - 1:
            maxc = -1
            maxcnt = 0
            for key, val in cnts.items():
                if val > maxcnt:
                    maxcnt = val
                    maxc = key
            s_nodes = [
                n for n in graph.node if graph.node[n]["cluster"] == maxc
            ]
            s_graph = graph.subgraph(s_nodes)
            edgecuts, parts = metis.part_graph(
            s_graph, 2, objtype="cut", ufactor=250, seed=42
        )
            new_part_cnt = 0
            for i, p in enumerate(s_graph.nodes()):
                if parts[i] == 1:
                    graph.node[p]["cluster"] = clusters + 1
                    new_part_cnt = new_part_cnt + 1
            if plotting is True:

                self.plot2d_graph_gui(
                    graph,
                    canvas=canvas,
                    ax=ax,
                    save_plots=self.save_plots,
                    ind_fig=self.ind_fig,
                    print_clust=False,
                )
                self.ind_fig += 1
            cnts[maxc] = cnts[maxc] - new_part_cnt
            cnts[clusters + 1] = new_part_cnt
            clusters = clusters + 1

        # edgecuts, parts = metis.part_graph(graph, k)
        if df is not None:
            df["cluster"] = nx.get_node_attributes(graph, "cluster").values()
        return graph

    def plot2d_graph_gui(
        self, graph, canvas, ax, save_plots, ind_fig=None, print_clust=True
    ):

        if self.delay != 0:
            pause_execution(self.delay)

        ax.clear()
        ax.set_title(self.name + " Graph Clustering")

        pos = nx.get_node_attributes(graph, "pos")
        colors = {
            0: "seagreen",
            1: "beige",
            2: "yellow",
            3: "grey",
            4: "pink",
            5: "turquoise",
            6: "orange",
            7: "purple",
            8: "yellowgreen",
            9: "olive",
            10: "brown",
            11: "tan",
            12: "plum",
            13: "rosybrown",
            14: "lightblue",
            15: "khaki",
            16: "gainsboro",
            17: "peachpuff",
            18: "lime",
            19: "peru",
            20: "dodgerblue",
            21: "teal",
            22: "royalblue",
            23: "tomato",
            24: "bisque",
            25: "palegreen",
        }

        el = nx.get_node_attributes(graph, "cluster").values()
        cmc = Counter(el).most_common()
        c = [colors[i % len(colors)] for i in el]

        if print_clust is True:
            self.log.appendPlainText("clusters: {}".format(cmc))

        if len(el) != 0:  # is set
            # print(pos)
            nx.draw(
                graph,
                pos,
                node_color=c,
                node_size=60,
                edgecolors="black",
                ax=ax,
            )
        else:
            nx.draw(graph, pos, node_size=60, edgecolors="black", ax=ax)

        canvas.draw()

        if save_plots is True:
            canvas.figure.savefig(
                "./Images/{}_{:02}/fig_{:02}.png".format(
                    self.name, self.ind_run, ind_fig
                )
            )

        QCoreApplication.processEvents()

    def plot2d_data_gui(
        self, df, canvas, ax, save_plots, ind_fig=None, col_i=None
    ):

        if self.delay != 0:
            pause_execution(self.delay)

        ax.clear()
        ax.set_title(self.name + " Merging")

        colors = {
            0: "seagreen",
            1: "dodgerblue",
            2: "yellow",
            3: "grey",
            4: "pink",
            5: "turquoise",
            6: "orange",
            7: "purple",
            8: "yellowgreen",
            9: "olive",
            10: "brown",
            11: "tan",
            12: "plum",
            13: "rosybrown",
            14: "lightblue",
            15: "khaki",
            16: "gainsboro",
            17: "peachpuff",
            18: "lime",
            19: "peru",
            20: "beige",
            21: "teal",
            22: "royalblue",
            23: "tomato",
            24: "bisque",
            25: "palegreen",
        }

        color_list = [colors[i] for i in df["cluster"]]

        df.plot(kind="scatter", c=color_list, x=0, y=1, ax=ax, s=100)

        ax.set_xlabel("")
        ax.set_ylabel("")

        if col_i is not None:
            ax.scatter(
                df[df.cluster == col_i].iloc[:, 0],
                df[df.cluster == col_i].iloc[:, 1],
                color="black",
                s=140,
                edgecolors="white",
                alpha=0.8,
            )

        canvas.draw()

        if save_plots is True:
            canvas.figure.savefig(
                "./Images/{}_{:02}/fig_{:02}.png".format(
                    self.name, self.ind_run, ind_fig
                )
            )

        QCoreApplication.processEvents()
