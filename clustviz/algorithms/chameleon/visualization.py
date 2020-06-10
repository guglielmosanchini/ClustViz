import networkx as nx
import matplotlib.pyplot as plt
from collections import Counter


def plot2d_graph(graph, print_clust=True):
    pos = nx.get_node_attributes(graph, "pos")
    colors = {
        0: "seagreen",
        1: "lightcoral",
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
        print("clusters: ", cmc)

    if len(el) != 0:  # is set
        # print(pos)
        nx.draw(graph, pos, node_color=c, node_size=60, edgecolors="black")
    else:
        nx.draw(graph, pos, node_size=60, edgecolors="black")
    plt.show(block=False)


def plot2d_data(df, col_i=None):
    if len(df.columns) > 3:
        print("Plot Warning: more than 2 dimensions!")

    df.plot(kind="scatter", c=df["cluster"], cmap="gist_rainbow", x=0, y=1)
    plt.xlabel("x")
    plt.ylabel("y")

    if col_i is not None:
        plt.scatter(
            df[df.cluster == col_i].iloc[:, 0],
            df[df.cluster == col_i].iloc[:, 1],
            color="black",
            s=120,
            edgecolors="white",
            alpha=0.8,
        )

    plt.show(block=False)
