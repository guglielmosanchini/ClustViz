import pandas as pd
import matplotlib.pyplot as plt
from collections import OrderedDict
from algorithms.optics import dist1
import random


def scan_neigh1_mod(data, point, eps):
    """
    Neighborhood search for a point of a given dataset-dictionary (data)
    with a fixed eps; it returns also the point itself, differently from
    scan_neigh1 of OPTICS.

    :param data: input dictionary.
    :param point: point whose neighborhood is to be examined.
    :param eps: radius of search.
    :return: dictionary of neighborhood points.
    """

    neigh = {}

    for i, element in enumerate(data.values()):

        d = dist1(element, point)

        if (d <= eps):

            neigh.update({str(i):element})

    return neigh



def point_plot_mod(X, X_dict, x,y, eps, ClustDict):
    """
    Plots a scatter plot of points, where the point (x,y) is light black and
    surrounded by a red circle of radius eps, where already processed point are plotted
    according to ClustDict and without edgecolor, whereas still-to-process points are green
    with black edgecolor.

    :param X: input array.
    :param X_dict: input dictionary version of X.
    :param x: x-coordinate of the point that is currently inspected.
    :param y: y-coordinate of the point that is currently inspected.
    :param eps: radius of the circle to plot around the point (x,y).
    :param ClustDict: dictionary of the form point_index:cluster_label, built by DBSCAN
    """

    colors = {-1:'red', 0:'lightblue', 1:'beige', 2:'yellow', 3:'grey',
              4:'pink', 5:'navy', 6:'orange', 7:'purple', 8:'salmon', 9:'olive', 10:'brown',
             11:'tan', 12: 'lime'}

    fig, ax = plt.subplots(figsize=(14,6))

    plt.scatter(X[:,0], X[:,1], s=300, color="lime", edgecolor="black")

    for i in ClustDict:
        plt.scatter(X_dict[i][0],X_dict[i][1], color=colors[ClustDict[i]%12], s=300 )

    plt.scatter(x=x,y=y,s=400, color="black", alpha=0.4)

    circle1 = plt.Circle((x, y), eps, color='r', fill=False, linewidth=3, alpha=0.7)
    plt.gcf().gca().add_artist(circle1)

    xmin, xmax, ymin, ymax = plt.axis()
    xwidth = xmax - xmin
    ywidth = ymax - ymin

    xw1 = xwidth*0.008
    yw1 = ywidth*0.01

    xw2 = xwidth*0.005
    yw2 = ywidth*0.01

    xw3 = xwidth*0.01
    yw3 = ywidth*0.01

    for i, txt in enumerate([i for i in range(len(X))]):
        if len(str(txt))==2:
            ax.annotate(txt, (X[:,0][i]-xw1, X[:,1][i]-yw1), fontsize=12, size=12)
        elif len(str(txt))==1:
            ax.annotate(txt, (X[:,0][i]-xw2, X[:,1][i]-yw2), fontsize=12, size=12)
        else:
            ax.annotate(txt, (X[:,0][i]-xw3, X[:,1][i]-yw3), fontsize=9, size=9)

    plt.show()



def plot_clust_DB(X, ClustDict, eps, circle_class=None, noise_circle=True):
    """
    Scatter plot of the data points, colored according to the cluster they belong to; circle_class Plots
    circles around some or all points, with a radius of eps; if Noise_circle is True, circle are also plotted
    around noise points.

    :param X: input array.
    :param ClustDict: dictionary of the form point_index:cluster_label, built by DBSCAN.
    :param eps: radius of the circles to plot around the points.
    :param circle_class: if == "true", plots circles around every non-noise point, else plots circles
                         only around points belonging to certain clusters, e.g. circle_class = [1,2] will
                         plot circles around points belonging to clusters 1 and 2.
    :param noise_circle: if True, plots circles around noise points

    """

    X_dict = dict(zip([str(i) for i in range(len(X))], X))

    new_dict = {key: (val1, ClustDict[key]) for key,val1 in zip(list(X_dict.keys()),list(X_dict.values())) }

    new_dict = OrderedDict((k, new_dict[k]) for k in list(ClustDict.keys()))

    df = pd.DataFrame(dict(x = [i[0][0] for i in list(new_dict.values())],
                 y=[i[0][1] for i in list(new_dict.values())],
                 label= [i[1]for i in list(new_dict.values())]), index=new_dict.keys())

    colors = {-1:'red', 0:'lightblue', 1:'beige', 2:'yellow', 3:'grey',
              4:'pink', 5:'navy', 6:'orange', 7:'purple', 8:'salmon', 9:'olive', 10:'brown',
             11:'tan', 12: 'lime'}

    fig, ax1 = plt.subplots(1, 1, figsize=(18,6))

    grouped = df.groupby('label')

    lista_lab = list(df.label.value_counts().index)

    for lab in lista_lab:

        df_sub = df[df.label == lab]
        plt.scatter(df_sub.x, df_sub.y, color = colors[lab%12], s=300, edgecolor="black")

    if noise_circle == True:

        df_noise = df[df.label == -1]

        for i in range(len(df_noise)):

            ax1.add_artist(plt.Circle((df_noise["x"].iloc[i],
                                       df_noise["y"].iloc[i]), eps, color='r', fill=False, linewidth=3, alpha=0.7))

    if circle_class is not None:

        if circle_class != "true":

            lista_lab = circle_class

        for lab in lista_lab:

            if lab != -1:

                df_temp = df[df.label == lab]

                for i in range(len(df_temp)):

                    ax1.add_artist(plt.Circle((df_temp["x"].iloc[i], df_temp["y"].iloc[i]),
                                              eps, color=colors[lab], fill=False, linewidth=3, alpha=0.7))

    ax1.set_xlabel("")
    ax1.set_ylabel("")


    xmin, xmax, ymin, ymax = plt.axis()
    xwidth = xmax - xmin
    ywidth = ymax - ymin

    xw1 = xwidth*0.005
    yw1 = ywidth*0.01

    xw2 = xwidth*0.0025
    yw2 = ywidth*0.01

    xw3 = xwidth*0.01
    yw3 = ywidth*0.01

    for i, txt in enumerate([i for i in range(len(X))]):
        if len(str(txt))==2:
            ax1.annotate(txt, (X[:,0][i]-xw1, X[:,1][i]-yw1), fontsize=8, size=10)
        elif len(str(txt))==1:
            ax1.annotate(txt, (X[:,0][i]-xw2, X[:,1][i]-yw2), fontsize=8, size=10)
        else:
            ax1.annotate(txt, (X[:,0][i]-xw3, X[:,1][i]-yw3), fontsize=6, size=8)


    plt.show()


def DBSCAN(data, eps, minPTS, plotting=False, print_details=False):
    """
    DBSCAN algorithm.

    :param data: input array.
    :param eps: radius of a point within which to search for minPTS points.
    :param minPTS: minimum number of neighbors for a point to be considered a core point.
    :param plotting: if True, executes point_plot_mod, plotting every time a points is
                     added to a clusters
    :param print_details: if True, prints the length of the "external" NearestNeighborhood
                          and of the "internal" one (in the while loop).
    :return ClustDict: dictionary of the form point_index:cluster_label.

    """

    ClustDict = {}

    clust_id = -1

    X_dict = dict(zip([str(i) for i in range(len(data))], data))

    processed = []

    processed_list = []

    #unprocessed = list(set(list(X_dict.keys())) - set(processed))

    for point in X_dict:

        if point not in processed:

            processed.append(point)

            #print(processed)

            N = scan_neigh1_mod(X_dict, X_dict[point], eps)

            if print_details == True:

                print("len(N): ", len(N))

            if len(N) < minPTS:

                ClustDict.update({point: -1})

                if plotting == True:

                    point_plot_mod(data, X_dict, X_dict[point][0], X_dict[point][1], eps, ClustDict, -1)

            else:

                clust_id+=1

                ClustDict.update({point: clust_id})

                if plotting == True:

                    point_plot_mod(data, X_dict, X_dict[point][0], X_dict[point][1], eps, ClustDict)

                processed_list = [point]

                del N[point]

                while len(N)>0:

                    if print_details == True:

                        print("len(N) in while loop: ", len(N))

                    n = random.choice(list(N.keys()))

                    while (n in processed_list):

                        n = random.choice(list(N.keys()))

                    processed_list.append(n)

                    del N[n]

                    if n not in processed:

                        processed.append(n)

                        N_2 = scan_neigh1_mod(X_dict, X_dict[n], eps)

                        if print_details == True:

                            print("len N2: ", len(N_2))

                        if len(N_2) >= minPTS:

                            for element in N_2:

                                if element not in processed_list:

                                    N.update({element: X_dict[element]})

                    if (n not in ClustDict) or (ClustDict[n] == -1):

                        ClustDict.update({n: clust_id})

                    if plotting == True:

                            point_plot_mod(data, X_dict, X_dict[n][0], X_dict[n][1], eps, ClustDict)

    return ClustDict
