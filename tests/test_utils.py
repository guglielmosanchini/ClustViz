from clustviz.utils import flatten_list, encircle, convert_colors
import matplotlib.pyplot as plt


def test_flatten_list():
    input_list = [[2], [3, 4], [5], ["a", 3, 7, 9]]

    flat_list = flatten_list(input_list)

    target_list = [2, 3, 4, 5, "a", 3, 7, 9]

    assert flat_list == target_list


def test_encircle(monkeypatch):
    fig, ax = plt.subplots(figsize=(14, 6))
    monkeypatch.setattr(plt, "show", lambda: None)
    X_clust = [1, 2, 3]
    Y_clust = [7, 3, 5]
    encircle(X_clust, Y_clust, ax)


def test_convert_colors():
    color_dict = {
        0: "red",
        1: "blue",
    }
    color_dict_rect = convert_colors(color_dict, alpha=0.3)

    assert color_dict_rect == {0: (1.0, 0.0, 0.0, 0.3), 1: (0.0, 0.0, 1.0, 0.3)}
