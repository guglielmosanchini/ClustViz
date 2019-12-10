import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from algorithms.optics import dist1
from algorithms.agglomerative import dist_mat_gen
from matplotlib.patches import Rectangle
from collections import Counter, OrderedDict
from copy import deepcopy
import random
import math



def point_plot_mod2(X, a, reps, level_txt, level2_txt=None, plot_lines=False,
                    par_index=None, u=None, u_cl=None, initial_ind=None, last_reps=None,
                    not_sampled=None, not_sampled_ind=None, n_rep_fin=None):

    if par_index is not None:
        diz = dict(zip(par_index,[i for i in range(len(par_index))]))

    fig, ax = plt.subplots(figsize=(14,6))

    plt.scatter(X[:,0], X[:,1], s=300, color="lime", edgecolor="black")

    a = a.dropna(1, how="all")

    colors = { 0:"seagreen", 1:'beige', 2:'yellow', 3:'grey',
               4:'pink', 5:'turquoise', 6:'orange', 7:'purple', 8:'yellowgreen', 9:'olive', 10:'brown',
               11:'tan', 12: 'plum', 13:'rosybrown', 14:'lightblue', 15:"khaki", 16:"gainsboro", 17:"peachpuff"}

    len_ind = [len(i.split("-")) for i in list(a.index)]
    start = np.min([i for i in range(len(len_ind)) if len_ind[i]>1])

    for ind,i in enumerate(range(start,len(a))):
        point = a.iloc[i].name.replace("(","").replace(")","").split("-")
        if par_index is not None:
            for j in range(len(point)):
                plt.scatter(X[diz[point[j]],0], X[diz[point[j]],1], s=350, color=colors[ind%18])
        else:
            point = [int(i) for i in point]
            for j in range(len(point)):
                plt.scatter(X[point[j],0], X[point[j],1], s=350, color=colors[ind%18])

    point = a.iloc[-1].name.replace("(","").replace(")","").split("-")
    #print(point)
    if par_index is not None:
        point = [diz[point[i]] for i in range(len(point))]
        com = X[point].mean(axis=0)
    else:
        point = [int(i) for i in point]
        com = X[point].mean(axis=0)

    plt.scatter(com[0], com[1], s=400, color="r", marker="X", edgecolor="black")

    #print(len(reps))
    x_reps = [i[0] for i in reps]
    y_reps = [i[1] for i in reps]
    plt.scatter(x_reps, y_reps, s=360, color="r", edgecolor="black")

    if par_index is not None:
        rect_min = X[point].min(axis=0)
        rect_diff = X[point].max(axis=0)-rect_min
    else:
        rect_min = X[point].min(axis=0)
        rect_diff = X[point].max(axis=0)-rect_min

    xmin, xmax, ymin, ymax = plt.axis()
    xwidth = xmax - xmin
    ywidth = ymax - ymin

    xx_rect = xwidth*0.015 #.05
    #yy_rect = ywidth #.1
    yy_rect = xx_rect*2

    if plot_lines == True:
        plt.axvline(rect_min[0], color="black")
        plt.axvline(rect_diff[0]+rect_min[0], color="black")
        plt.axhline(rect_min[1], color="black")
        plt.axhline(rect_diff[1]+rect_min[1], color="black")


    plt.gcf().gca().add_patch(Rectangle((rect_min[0] - xx_rect, rect_min[1] - xx_rect),
                                        rect_diff[0]+yy_rect, rect_diff[1]+yy_rect, fill=True,
                                        color=colors[ind%18], alpha=0.1, linewidth=3,
                                        ec="black"))
    plt.gcf().gca().add_patch(Rectangle((rect_min[0] - xx_rect, rect_min[1] - xx_rect),
                                        rect_diff[0]+yy_rect, rect_diff[1]+yy_rect, fill=None,
                                        color='r', alpha=1, linewidth=3
                                        ))

    xw1 = xwidth*0.009
    yw1 = ywidth*0.015

    xw2 = xwidth*0.005
    yw2 = ywidth*0.01

    xw3 = xwidth*0.01
    yw3 = ywidth*0.01

    if initial_ind is not None:
        for i, txt in enumerate(initial_ind):
            if len(str(txt))==2:
                ax.annotate(txt, (X[:,0][i]-xw1, X[:,1][i]-yw1), fontsize=12, size=12)
            elif len(str(txt))==1:
                ax.annotate(txt, (X[:,0][i]-xw2, X[:,1][i]-yw2), fontsize=12, size=12)
            else:
                ax.annotate(txt, (X[:,0][i]-xw3, X[:,1][i]-yw3), fontsize=9, size=9)

    else:
        for i, txt in enumerate([i for i in range(len(X))]):
            if len(str(txt))==2:
                ax.annotate(txt, (X[:,0][i]-xw1, X[:,1][i]-yw1), fontsize=12, size=12)
            elif len(str(txt))==1:
                ax.annotate(txt, (X[:,0][i]-xw2, X[:,1][i]-yw2), fontsize=12, size=12)
            else:
                ax.annotate(txt, (X[:,0][i]-xw3, X[:,1][i]-yw3), fontsize=9, size=9)

    ax.annotate("min_dist: " + str(round(level_txt,5)), (xmax*0.75,ymax*0.9), fontsize=12, size=12)

    if level2_txt is not None:
        ax.annotate("dist_incr: " + str(round(level2_txt,5)), (xmax*0.75,ymax*0.8), fontsize=12, size=12)

    ax.annotate("n° clust: " + str(len(a)), (xmax*0.75,ymax*0.7), fontsize=12, size=12)

    plt.show()


    if last_reps is not None:

        fig, ax = plt.subplots(figsize=(14,6))

        plt.scatter(X[:,0], X[:,1], s=300, color="lime", edgecolor="black")

        coms = []
        for ind,i in enumerate(range(0,len(a))):
            point = a.iloc[i].name.replace("(","").replace(")","").split("-")
            for j in range(len(point)):
                plt.scatter(X[diz[point[j]],0], X[diz[point[j]],1], s=350, color=colors[ind%18])
            point = [diz[point[i]] for i in range(len(point))]
            coms.append(X[point].mean(axis=0))


        colors_reps = ["red", "crimson","indianred", "lightcoral", "salmon", "darksalmon", "firebrick"]

        flat_reps = [item for sublist in list(last_reps.values()) for item in sublist]

        for i in range(len(last_reps)):
            len_rep = len(list(last_reps.values())[i])

            x = [list(last_reps.values())[i][j][0] for j in range(min(n_rep_fin, len_rep))]
            y = [list(last_reps.values())[i][j][1] for j in range(min(n_rep_fin, len_rep))]

            plt.scatter(x, y, s=400, color=colors_reps[i], edgecolor="black")
            plt.scatter(coms[i][0], coms[i][1], s=400, color=colors_reps[i], marker="X", edgecolor="black")

            for num in range(min(n_rep_fin, len_rep)):
                plt.gcf().gca().add_artist(plt.Circle((x[num], y[num]), xwidth*0.03,
                                                      color=colors_reps[i], fill=False, linewidth=3, alpha=0.7))

            plt.scatter(not_sampled[:,0], not_sampled[:,1], s=400, color="lime", edgecolor="black")

        for ind in range(len(not_sampled)):
            dist_int = []
            for el in flat_reps:
                dist_int.append(dist1(not_sampled[ind], el))
            ind_min = np.argmin(dist_int)

            plt.arrow(not_sampled[ind][0],not_sampled[ind][1],
                      flat_reps[ind_min][0]-not_sampled[ind][0], flat_reps[ind_min][1]-not_sampled[ind][1],
                      length_includes_head=True, head_width=0.03, head_length=0.05)

        for i, txt in enumerate(initial_ind):
            if len(str(txt))==2:
                ax.annotate(txt, (X[:,0][i]-xw1, X[:,1][i]-yw1), fontsize=12, size=12)
            elif len(str(txt))==1:
                ax.annotate(txt, (X[:,0][i]-xw2, X[:,1][i]-yw2), fontsize=12, size=12)
            else:
                ax.annotate(txt, (X[:,0][i]-xw3, X[:,1][i]-yw3), fontsize=9, size=9)

        if not_sampled_ind is not None:
            for i, txt in enumerate(not_sampled_ind):
                if len(str(txt))==2:
                    ax.annotate(txt, (not_sampled[:,0][i]-xw1, not_sampled[:,1][i]-yw1), fontsize=12, size=12)
                elif len(str(txt))==1:
                    ax.annotate(txt, (not_sampled[:,0][i]-xw2, not_sampled[:,1][i]-yw2), fontsize=12, size=12)
                else:
                    ax.annotate(txt, (not_sampled[:,0][i]-xw3, not_sampled[:,1][i]-yw3), fontsize=9, size=9)

        plt.show()

    if par_index is not None:

        diz["(" + u + ")" + "-" + "(" + u_cl + ")"] = len(diz)
        list_keys_diz = list(diz.keys())

        return list_keys_diz



def dist_clust_cure(rep_u, rep_v):
    rep_u = np.array(rep_u)
    rep_v = np.array(rep_v)
    distances = []
    for i in rep_u:
        for j in rep_v:
            #print(i,j)
            distances.append(dist1(i,j))
    return np.min(distances)



def update_mat_cure(mat,i,j, rep_new, name):

    a1 = mat.loc[i]
    b1 = mat.loc[j]

    key_lists = list(rep_new.keys())
    #print(len(key_lists))

    vec = []
    for i in range(len(mat)):
        #print(i)
        vec.append(dist_clust_cure(rep_new[name], rep_new[key_lists[i]]))


    mat.loc["("+a1.name+")"+"-"+"("+b1.name+")",:] = vec
    mat["("+a1.name+")"+"-"+"("+b1.name+")"] = vec + [np.inf]

    mat = mat.drop([a1.name,b1.name], 0)
    mat = mat.drop([a1.name,b1.name], 1)

    return mat


def sel_rep(clusters, name, c, alpha):

    if len(clusters[name]) <= c:

        others = clusters[name]
        com = np.mean(clusters[name], axis=0)

        for i in range(len(others)):
            others[i] = others[i] + alpha*(com - others[i])

        return others

    else:

        others = []
        indexes = []

        points = clusters[name]
        com = np.mean(points, axis=0)

        distances_com = {i : dist1(points[i], com) for i in range(len(points))}
        index = max(distances_com, key=distances_com.get)

        indexes.append(index)
        others.append(np.array(points[index])) # first point

        for step in range(min(c-1,len(points)-1)):
            partial_distances = {str(i): [] for i in range(len(points))}
            for i in range(len(points)):
                if i not in indexes:
                    for k in range(len(others)):
                        #print(partial_distances)
                        partial_distances[str(i)].append([dist1(points[i], np.array(others[k]))])
            partial_distances = dict((k, [np.sum(v)]) for k,v in partial_distances.items())
            index2 = max(partial_distances, key=partial_distances.get)
            indexes.append(int(index2))
            others.append(points[int(index2)])

        for i in range(len(others)):
            others[i] = others[i] + alpha*(com - others[i])

        return others



def sel_rep_fast(prec_reps, clusters, name, c, alpha):

    com = np.mean(clusters[name], axis=0)

    if len(prec_reps) <= c:

        others = prec_reps
        for i in range(len(others)):
            others[i] = others[i] + alpha*(com - others[i])

        return others

    else:

        others = []
        indexes = []

        points = prec_reps

        distances_com = {i : dist1(points[i], com) for i in range(len(points))}
        index = max(distances_com, key=distances_com.get)

        indexes.append(index)
        others.append(np.array(points[index])) # first point

        for step in range(min(c-1,len(points)-1)):
            partial_distances = {str(i): [] for i in range(len(points))}
            for i in range(len(points)):
                if i not in indexes:
                    for k in range(len(others)):
                        #print(partial_distances)
                        partial_distances[str(i)].append([dist1(points[i], np.array(others[k]))])
            partial_distances = dict((k, [np.sum(v)]) for k,v in partial_distances.items())
            index2 = max(partial_distances, key=partial_distances.get)
            indexes.append(int(index2))
            others.append(points[int(index2)])

        for i in range(len(others)):
            others[i] = others[i] + alpha*(com - others[i])

        return others




def cure(X, k, c=3, alpha=0.1, plotting=True, preprocessed_data = None,
         partial_index = None, n_rep_finalclust=None, not_sampled=None, not_sampled_ind=None):

    if preprocessed_data is None:
        l = [[i,i] for i in range(len(X))]
        flat_list = [item for sublist in l for item in sublist]
        col = [str(el)+"x" if i%2==0 else str(el)+"y" for i, el in enumerate(flat_list)]

        if partial_index is not None:
            a = pd.DataFrame(index = partial_index, columns = col)
        else:
            a = pd.DataFrame(index = [str(i) for i in range(len(X))], columns = col)

        a["0x"]=X.T[0]
        a["0y"]=X.T[1]

        b = a.dropna(axis=1, how="all")

        # initial clusters
        #clusters = {str(i):X[i] for i in range(len(X))}
        if partial_index is not None:
            clusters = dict(zip(partial_index, X))
        else:
            clusters = {str(i): np.array(X[i]) for i in range(len(X))}

        # build Xdist
        X_dist1 = dist_mat_gen(b)

        #initialize representatives
        if partial_index is not None:
            rep = {partial_index[i]: [X[int(i)]] for i in range(len(X))}
        else:
            rep = {str(i): [X[i]] for i in range(len(X))}

        #just as placeholder for while loop
        heap = [1]*len(X_dist1)

        # store minimum distances between clusters for each iteration
        levels = []

    else:

        clusters = preprocessed_data[0]
        rep = preprocessed_data[1]
        a = preprocessed_data[2]
        X_dist1 = preprocessed_data[3]
        heap = [1]*len(X_dist1)
        levels = []

    if partial_index is not None:
        initial_index = deepcopy(partial_index)

    while len(heap) > k:
        #print(len(heap))

        list_argmin = list(X_dist1.apply(lambda x: np.argmin(x)).values)
        list_min = list(X_dist1.min(axis=0).values)
        heap = dict(zip(list(X_dist1.index), list_min))
        heap = dict(OrderedDict(sorted(heap.items(), key=lambda kv: kv[1])))
        closest = dict(zip(list(X_dist1.index), list_argmin))

        #get minimum key and delete it
        u = min(heap, key=heap.get)
        levels.append(heap[u])
        del heap[u]
        u_cl = closest[u]
        del closest[u]

        #form the new cluster
        if (np.array(clusters[u]).shape == (2,)) and (np.array(clusters[u_cl]).shape == (2,)):
            w = [clusters[u],clusters[u_cl]]
        elif (np.array(clusters[u]).shape != (2,)) and (np.array(clusters[u_cl]).shape == (2,)):
            clusters[u].append(clusters[u_cl])
            w = clusters[u]
        elif (np.array(clusters[u]).shape == (2,)) and (np.array(clusters[u_cl]).shape != (2,)):
            clusters[u_cl].append(clusters[u])
            w = clusters[u_cl]
        else:
            w = clusters[u] + clusters[u_cl]

        #delete old cluster
        del clusters[u]
        del clusters[u_cl]

        name = "(" + u + ")" + "-" + "(" + u_cl + ")"
        clusters[name] = w

        #update representatives
        rep[name] = sel_rep_fast(rep[u] + rep[u_cl], clusters, name, c, alpha)

        #update distance matrix
        X_dist1 = update_mat_cure(X_dist1, u, u_cl, rep, name)

        del rep[u]
        del rep[u_cl]

        if plotting == True:

            dim1 = int(a.loc[u].notna().sum())

            a.loc["("+u+")"+"-"+"("+u_cl+")",:] = a.loc[u].fillna(0) + a.loc[u_cl].shift(dim1, fill_value=0)
            a = a.drop(u,0)
            a = a.drop(u_cl,0)

            if partial_index is not None:

                if (len(heap) == k) and (not_sampled is not None) and (not_sampled_ind is not None):
                    #print(rep)
                    final_reps = {list(rep.keys())[i] : random.sample(list(rep.values())[i],
                                                                      min(n_rep_finalclust,len(list(rep.values())[i]))) for i in range(len(rep))}
                    partial_index = point_plot_mod2(X=X, a=a, reps=rep[name],
                                                level_txt=levels[-1], par_index=partial_index,
                                                u=u, u_cl=u_cl, initial_ind = initial_index,
                                                last_reps= final_reps, not_sampled=not_sampled,
                                                not_sampled_ind = not_sampled_ind, n_rep_fin=n_rep_finalclust)
                else:
                    partial_index = point_plot_mod2(X=X, a=a, reps=rep[name],
                                                level_txt=levels[-1], par_index=partial_index,
                                                u=u, u_cl=u_cl, initial_ind = initial_index)
            else:
                point_plot_mod2(X, a, rep[name], levels[-1])

    return (clusters, rep, a)



def plot_results_cure(clust):
    """
    Scatter plot of data points, colored according to the cluster they belong to, after performing
    CURE algorithm.

    :param clust: output of CURE algorithm, dictionary of the form cluster_labels+point_indices: coords of points
    """
    cl_list = []
    for num_clust in range(len(clust)):
        cl_list.append(np.array(clust[list(clust.keys())[num_clust]]))
        try:
            plt.scatter(cl_list[-1][:,0], cl_list[-1][:,1])
        except:
            plt.scatter(cl_list[-1][0], cl_list[-1][1])
    plt.show()




def Chernoff_Bounds(u_min, f, N, d, k):
    """
    u_min: size of the smallest cluster u,
    f: percentage of cluster points (0 <= f <= 1),
    N: total size,
    s: sample size,
    d: 0 <= d <= 1
    the probability that the sample contains less than f*|u| points of cluster u is less than d

    if one uses as |u| the minimum cluster size we are interested in, the result is
    the minimum sample size that guarantees that for k clusters
    the probability of selecting fewer than f*|u| points from any one of the clusters u is less than k*d.

    """

    l = np.log(1/d)
    res = f*N + N/u_min * l + N/u_min * np.sqrt(l**2 + 2*f*u_min*l)
    print("If the sample size is {0}, the probability of selecting fewer than {1} points from".format(math.ceil(res),round(f*u_min)) \
          + " any one of the clusters is less than {0}".format(k*d))

    return res


def dist_mat_gen_cure(dictionary):

    #even_num = [i for i in range(2,len(X)+1) if i%2==0]
    D = pd.DataFrame()
    ind = list(dictionary.keys())
    k = 0
    for i in ind:
        for j in ind[k:]:
            if i!=j:

                a = dictionary[i]
                b = dictionary[j]

                D.loc[i,j] = dist_clust_cure(a, b)
                D.loc[j,i] = D.loc[i,j]
            else:

                D.loc[i,j] = np.inf

        k += 1

    D = D.fillna(np.inf)

    return D


def cure_sample_part(X, k, c=3, alpha=0.3, u_min=None, f=0.3, d=0.02, p=None, q=None, n_rep_finalclust=None):

    if u_min is None:
        u_min = round(len(X)/k)

    if n_rep_finalclust is None:
        n_rep_finalclust=c

    l = [[i,i] for i in range(len(X))]
    flat_list = [item for sublist in l for item in sublist]
    col = [str(el)+"x" if i%2==0 else str(el)+"y" for i, el in enumerate(flat_list)]
    a = pd.DataFrame(index=[str(i) for i in range(len(X))], columns=col)
    a["0x"]=X.T[0]
    a["0y"]=X.T[1]
    b = a.dropna(axis=1, how="all")

    n = math.ceil(Chernoff_Bounds(u_min=u_min, f=f, N=len(X), k=k, d=d))
    b_sampled = b.sample(n, random_state=42)
    b_notsampled = b.loc[[str(i) for i in range(len(b)) if str(i) not in b_sampled.index], :]

    if (p is None) and (q is None):

        def g(x):
            res = (x[1]-1)/(x[0]*x[1]) + 1/(x[1]**2)
            return res

        results = {}
        for i in range(2,15):
            for j in range(2,15):
                results[(i,j)] = g([i,j])
        p, q = max(results, key=results.get)
        print("p: ", p)
        print("q: ", q)

    if (n/(p*q)) < 2*k:
        print("n/pq is less than 2k, results could be wrong")

    z = round(n/p)
    lin_sp = np.linspace(0,n,p+1, dtype="int")
    #lin_sp
    b_partitions = []
    for num_p in range(p):
        try:
            b_partitions.append(b_sampled.iloc[lin_sp[num_p]:lin_sp[num_p+1]])
        except:
            b_partitions.append(b_sampled.iloc[lin_sp[num_p]:])

    k_prov = round(n/(p*q))

    partial_clust1 = []
    partial_rep1 = []
    partial_a1 = []

    for i in range(p):
        print("\n")
        print(i)
        clusters, rep, mat_a = cure(b_partitions[i].values, k=k_prov, c=c, alpha=alpha, partial_index=b_partitions[i].index)
        partial_clust1.append(clusters)
        partial_rep1.append(rep)
        partial_a1.append(mat_a)

    #merging all data into single components
    #clusters
    clust_tot = {}
    for d in partial_clust1:
        clust_tot.update(d)
    #representatives
    rep_tot = {}
    for d in partial_rep1:
        rep_tot.update(d)
    #mat a
    diz = {i:len(b_partitions[i]) for i in range(p)}
    num_freq = Counter(diz.values()).most_common(1)[0][0]
    bad_ind = [list(diz.keys())[i] for i in range(len(diz)) if diz[i] != num_freq]

    for ind in bad_ind:
        partial_a1[ind]["{0}x".format(diz[ind])] = [np.nan]*k_prov
        partial_a1[ind]["{0}y".format(diz[ind])] = [np.nan]*k_prov

    for i in range(len(partial_a1)-1):
        if i == 0:
            a_tot = partial_a1[i].append(partial_a1[i+1])
        else:
            a_tot = a_tot.append(partial_a1[i+1])
    # mat Xdist
    X_dist_tot = dist_mat_gen_cure(rep_tot)

    # final_clustering
    prep_data = [clust_tot, rep_tot, a_tot, X_dist_tot]
    clusters, rep, mat_a = cure(b_sampled.values, k=k, c=c, alpha=alpha, preprocessed_data=prep_data,
                                partial_index=b_sampled.index, n_rep_finalclust=n_rep_finalclust, not_sampled=b_notsampled.values,
                                not_sampled_ind=b_notsampled.index)


def demo_parameters():
    """Four plots showing the effects on the sample size of various parameters"""

    plt.figure(figsize=(12,10))
    plt.suptitle("Effects on sample size from different parameters")

    ax0 = plt.subplot(2, 2, 1)
    #plt.plot(d, k*res)
    u_size=6000
    f=0.20
    N=20000
    k=4
    d = np.linspace(0.0000001,1,100)
    ax0.set_title("u_min: {0}, f:{1}, k:{2}".format(u_size, f, k))
    res = k*(f*N + N/u_size * np.log(1/d) + N/u_size * np.sqrt(np.log(1/d)**2 + 2*f*u_size*np.log(1/d)))
    plt.axhline(N, color="r")
    plt.plot(d, res)
    plt.xlabel("d")
    plt.ylabel("sample size")

    ax1 = plt.subplot(2, 2, 2)

    u_size=3000
    f=0.2
    N=20000
    d=0.1
    k = [1,2,3,4,5,6,7,8,9,10,11,12]
    ax1.set_title("u_min: {0}, f:{1}, d:{2}".format(u_size, f, d))
    res = [k[i]*(f*N + N/u_size * np.log(1/d) + N/u_size * np.sqrt(np.log(1/d)**2 + 2*f*u_size*np.log(1/d))) for i in range(len(k))]
    plt.axhline(N, color="r")
    plt.plot(k, res)
    plt.xlabel("k")

    ax2 = plt.subplot(2, 2, 3)

    u_size=5000
    f= np.linspace(0.00001,1,100)
    N=20000
    d=0.1
    k = 4
    ax2.set_title("u_min: {0}, d:{1}, k:{2}".format(u_size, d, k))
    res = k*(f*N + N/u_size * np.log(1/d) + N/u_size * np.sqrt(np.log(1/d)**2 + 2*f*u_size*np.log(1/d)))
    plt.axhline(N, color="r")
    plt.plot(f, res)
    plt.xlabel("f")
    plt.ylabel("sample size")

    ax3 = plt.subplot(2, 2, 4)

    u_size= np.linspace(200,10000,30)
    f= 0.2
    N=20000
    d=0.1
    k = 4
    ax3.set_title("f: {0}, d:{1}, k:{2}".format(f, d, k))
    res = k*(f*N + N/u_size * np.log(1/d) + N/u_size * np.sqrt(np.log(1/d)**2 + 2*f*u_size*np.log(1/d)))
    plt.axhline(N, color="r")
    plt.plot(u_size, res)
    plt.xlabel("u_min")

    plt.show()
