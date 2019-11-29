import random
import numpy as np
import matplotlib.pyplot as plt
from collections import OrderedDict
import pandas as pd



def point_plot(X, X_dict, x, y, eps, processed=None, col='yellow'):

    fig, ax = plt.subplots(figsize=(14,6))

    plt.scatter(X[:,0], X[:,1], s=300, color="lime", edgecolor="black")

    if processed is not None:
        for i in processed:
            plt.scatter(X_dict[i][0],X_dict[i][1], color=col, s=300 )

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

    for i, txt in enumerate([i for i in range(len(X))]):
        if len(str(txt))==2:
            ax.annotate(txt, (X[:,0][i]-xw1, X[:,1][i]-yw1), fontsize=12, size=12)
        else:
            ax.annotate(txt, (X[:,0][i]-xw2, X[:,1][i]-yw2), fontsize=12, size=12)

    plt.show()


#point_plot(X, dict(zip([str(i) for i in range(len(X))], X)), X[0,0], X[0,1], eps=0.3)


# original euclidean distance
def dist1(x,y):
    return np.sqrt(np.sum((x-y)**2))


# euclidean distance which takes keys of a dictionary (X_dict) as inputs
def dist2(data, x,y):
    return np.sqrt(np.sum((data[x]-data[y])**2))


# neighborhood search for a point of a given dataset-dictionary (data) with a fixed eps
def scan_neigh1(data, point, eps):

    neigh = {}
    distances = {}

    for i, element in enumerate(data.values()):

        d = dist1(element, point)

        if (d <= eps) and (d != 0):

            neigh.update({str(i):element})

    return neigh


# variation that returns only the keys of the input dictionary with the euclidean distances <= eps from the point
def scan_neigh2(data, point, eps):

    neigh = {}
    distances = {}

    for i, element in enumerate(data.values()):

        d = dist1(element, point)

        if (d <= eps) and (d != 0):

            distances.update({str(i):d})

    d_sorted = sorted(distances.items(), key=lambda x: x[1], reverse=False)

    return d_sorted


# returns the minPTS-distance of a point if it is a core point, else it returns np.inf
def minPTSdist(data, o, minPTS, eps):

    S = scan_neigh2(data, data[o], eps)

    if len(S) >= minPTS-1:

        return S[minPTS-2][1]

    else:

        return np.inf


#reachability distance (even if it is not a distance because it isn't symmetrical)
def reach_dist(data, x,y, minPTS, eps):

    return max(dist2(data, x,y), minPTSdist(data, y, minPTS, eps))



# plots the reachability plot, along with a horizontal line denoting eps, from the ClustDist produced by OPTICS
def Reach_plot(data, ClustDist, eps):
    plot_dic = {}

    for key, value in ClustDist.items():

        if np.isinf(value) == True:

            plot_dic[key] = eps*1.15

        else:

            plot_dic[key] = ClustDist[key]

    missing_keys = list(set(data.keys())-set(ClustDist.keys()))

    tick_list = list(ClustDist.keys()) + [' ']*(len(missing_keys))

    for m_k in missing_keys:

        plot_dic[m_k] = 0

    fig, ax = plt.subplots(1, 1, figsize=(12,5))

    ax.bar(plot_dic.keys(),plot_dic.values())

    ax.set_xticks(tick_list)

    ax.axhline(eps, color="red", linewidth=3)

    ax1 = ax.twinx()
    ax1.set_ylim(ax.get_ylim())
    ax1.set_yticks([eps])
    ax1.set_yticklabels(["\u03B5"])

    plt.show()



def OPTICS(X, eps, minPTS, plot=True, plot_reach=False):

    ClustDist = {}
    CoreDist = {}
    Seed =  {}
    processed = []

    # create dictionary
    X_dict = dict(zip([str(i) for i in range(len(X))], X))

    while len(processed) != len(X):

        if len(Seed) == 0:

                unprocessed = list(set(list(X_dict.keys())) - set(processed))

                (o, r) = (random.choice(unprocessed), np.inf)

        else:

            (o, r) = (min(Seed, key=Seed.get), Seed[min(Seed, key=Seed.get)])

            del Seed[o]

        N = scan_neigh1(X_dict, X_dict[o], eps)

        ClustDist.update({o:r})

        CoreDist.update({o: minPTSdist(X_dict, o, minPTS, eps)})

        if plot == True:

            point_plot(X, X_dict, X_dict[o][0], X_dict[o][1], eps, processed)

            if plot_reach == True:

                Reach_plot(X_dict, ClustDist, eps)


        processed.append(o)

        if len(N) >= minPTS-1 :

            for n in N:

                if n in processed:

                    continue

                else:

                    p = reach_dist(X_dict, n, o, minPTS, eps)

                    if n in Seed:

                        if p < Seed[n]:

                            Seed[n] = p
                    else:

                        Seed.update({n:p})

    return (ClustDist, CoreDist)



#extracts cluster in a DBSCAN fashion; one can use any eps_db <= eps of OPTICS
def ExtractDBSCANclust(ClustDist, CoreDist, eps_db):

    Clust_Dict = {}

    noise = -1

    clust_id = -1

    for key, value in ClustDist.items():

        if value > eps_db:

            if CoreDist[key] <= eps_db:

                clust_id += 1

                Clust_Dict[key] = clust_id

            else:

                Clust_Dict[key] = noise

        else:

            Clust_Dict[key] = clust_id

    return Clust_Dict


def plot_clust(X, ClustDist, CoreDist, eps, eps_db):

    X_dict = dict(zip([str(i) for i in range(len(X))], X))

    cl = ExtractDBSCANclust(ClustDist, CoreDist, eps_db)

    new_dict = {key: (val1, cl[key]) for key,val1 in zip(list(X_dict.keys()),list(X_dict.values())) }

    new_dict = OrderedDict((k, new_dict[k]) for k in list(ClustDist.keys()))

    df = pd.DataFrame(dict(x = [i[0][0] for i in list(new_dict.values())],
                 y=[i[0][1] for i in list(new_dict.values())],
                 label= [i[1]for i in list(new_dict.values())]), index=new_dict.keys())

    colors = {-1:'red', 0:'lightblue', 1:'beige', 2:'yellow', 3:'grey',
              4:'pink', 5:'navy', 6:'orange', 7:'purple', 8:'salmon', 9:'olive', 10:'brown',
             11:'tan', 12: 'lime'}

    #fig, ax = plt.subplots(figsize=(14,6))
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(18,6))

    grouped = df.groupby('label')
    for key, group in grouped:
        group.plot(ax=ax1, kind='scatter', x='x', y='y', label=key, color=colors[key], s=300, edgecolor="black")

    ax1.set_xlabel("")
    ax1.set_ylabel("")

    xmin, xmax, ymin, ymax = ax1.axis()
    xwidth = xmax - xmin
    ywidth = ymax - ymin

    xw1 = xwidth*0.01
    yw1 = ywidth*0.01

    xw2 = xwidth*0.003
    yw2 = ywidth*0.01

    for i, txt in enumerate([i for i in range(len(X))]):
        if len(str(txt))==2:
            ax1.annotate(txt, (X[:,0][i]-xw1, X[:,1][i]-yw1), fontsize=8, size=10)
        else:
            ax1.annotate(txt, (X[:,0][i]-xw2, X[:,1][i]-yw2), fontsize=8, size=10)


    #plt.show()

    plot_dic = {}

    for key, value in ClustDist.items():

        if np.isinf(value) == True:

            plot_dic[key] = eps*1.15

        else:

            plot_dic[key] = ClustDist[key]

    #plt.figure(figsize=(8,5), ax=ax)
    #fig, ax = plt.subplots(figsize=(8,5))
    ax2.bar(plot_dic.keys(),plot_dic.values(), color=[colors[i] for i in df.label], edgecolor="black")

    ax2.axhline(eps, color="black", linewidth=3)

    #plt.annotate("\u03B5", (0.920,0.79), xycoords='figure fraction')

    ax2.axhline(eps_db, color="black", linewidth=3)

    #plt.annotate("\u03B5"+"\'", (438,eps_db*103), xycoords='axes points')

    ax3 = ax2.twinx()
    ax3.set_ylim(ax2.get_ylim())
    ax3.set_yticks([eps, eps_db])
    ax3.set_yticklabels(["\u03B5", "\u03B5"+"\'"])

    plt.show()
