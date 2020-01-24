from sklearn.datasets.samples_generator import make_blobs, make_moons, make_circles


def choose_dataset(chosen_dataset, n_points):
    X = None

    if chosen_dataset == "blobs":
        X = make_blobs(n_samples=n_points, centers=4, n_features=2, cluster_std=1.5, random_state=42)[0]
    elif chosen_dataset == "moons":
        X = make_moons(n_samples=n_points, noise=0.05, random_state=42)[0]
    elif chosen_dataset == "scatter":
        X = make_blobs(n_samples=n_points, cluster_std=[2.5, 2.5, 2.5], random_state=42)[0]
    elif chosen_dataset == "circle":
        X = make_circles(n_samples=n_points, noise=0, random_state=42)[0]

    return X
