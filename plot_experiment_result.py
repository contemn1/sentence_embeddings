import matplotlib.pyplot as plt
import numpy as np
from matplotlib.pyplot import figure
from find_similar_sentences import get_final_result
import json
import matplotlib

def plot_bar_chart(result_dict):
    model_names = []
    add_means = []
    mult_means = []
    for key, value in result_dict.items():
        model_names.append(key)
        add_means.append(value[0])
        mult_means.append(value[1])

    figure(num=None, figsize=(14, 10.5), dpi=200, facecolor='w', edgecolor='k')
    ind = np.arange(len(model_names))
    width = 0.3
    plt.barh(ind, add_means[::-1], width, label='3CosAdd')
    plt.barh(ind + width, mult_means[::-1], width,
        label='3CosMul')

    plt.xlabel('Accuracy')
    plt.yticks(ind + width , model_names[::-1])
    plt.legend(loc='best')
    plt.savefig("/home/zxj/Data/relation_based_analogy/result/relation_analogy_plot")


def heatmap(data, row_labels, col_labels, ax=None,
            cbar_kw={}, cbarlabel="", **kwargs):
    """
    Create a heatmap from a numpy array and two lists of labels.

    Parameters
    ----------
    data
        A 2D numpy array of shape (N, M).
    row_labels
        A list or array of length N with the labels for the rows.
    col_labels
        A list or array of length M with the labels for the columns.
    ax
        A `matplotlib.axes.Axes` instance to which the heatmap is plotted.  If
        not provided, use current axes or create a new one.  Optional.
    cbar_kw
        A dictionary with arguments to `matplotlib.Figure.colorbar`.  Optional.
    cbarlabel
        The label for the colorbar.  Optional.
    **kwargs
        All other arguments are forwarded to `imshow`.
    """

    if not ax:
        ax = plt.gca()

    # Plot the heatmap
    im = ax.imshow(data, **kwargs)

    # Create colorbar
    '''
    cbar = ax.figure.colorbar(im, ax=ax, **cbar_kw)
    cbar.ax.set_ylabel(cbarlabel, rotation=-90, va="bottom")
    '''

    # We want to show all ticks...
    ax.set_xticks(np.arange(data.shape[1]))
    ax.set_yticks(np.arange(data.shape[0]))
    # ... and label them with the respective list entries.
    ax.set_xticklabels(col_labels)
    ax.set_yticklabels(row_labels)

    # Let the horizontal axes labeling appear on top.
    ax.tick_params(top=True, bottom=False,
                   labeltop=True, labelbottom=False)

    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=-30, ha="right",
             rotation_mode="anchor")

    # Turn spines off and create white grid.
    for edge, spine in ax.spines.items():
        spine.set_visible(False)

    ax.set_xticks(np.arange(data.shape[1]+1)-.5, minor=True)
    ax.set_yticks(np.arange(data.shape[0]+1)-.5, minor=True)
    ax.grid(which="minor", color="w", linestyle='-', linewidth=3)
    ax.tick_params(which="minor", bottom=False, left=False)

    return im


def annotate_heatmap(im, data=None, valfmt="{x:.2f}",
                     textcolors=["black", "white"],
                     threshold=None, **textkw):
    """
    A function to annotate a heatmap.

    Parameters
    ----------
    im
        The AxesImage to be labeled.
    data
        Data used to annotate.  If None, the image's data is used.  Optional.
    valfmt
        The format of the annotations inside the heatmap.  This should either
        use the string format method, e.g. "$ {x:.2f}", or be a
        `matplotlib.ticker.Formatter`.  Optional.
    textcolors
        A list or array of two color specifications.  The first is used for
        values below a threshold, the second for those above.  Optional.
    threshold
        Value in data units according to which the colors from textcolors are
        applied.  If None (the default) uses the middle of the colormap as
        separation.  Optional.
    **kwargs
        All other arguments are forwarded to each call to `text` used to create
        the text labels.
    """

    if not isinstance(data, (list, np.ndarray)):
        data = im.get_array()

    # Normalize the threshold to the images color range.
    if threshold is not None:
        threshold = im.norm(threshold)
    else:
        threshold = im.norm(data.max())/2.

    # Set default alignment to center, but allow it to be
    # overwritten by textkw.
    kw = dict(horizontalalignment="center",
              verticalalignment="center")
    kw.update(textkw)

    # Get the formatter in case a string is supplied
    if isinstance(valfmt, str):
        valfmt = matplotlib.ticker.StrMethodFormatter(valfmt)

    # Loop over the data and create a `Text` for each "pixel".
    # Change the text's color depending on the data.
    texts = []
    for i in range(data.shape[0]):
        for j in range(data.shape[1]):
            kw.update(color=textcolors[int(im.norm(data[i, j]) > threshold)])
            text = im.axes.text(j, i, valfmt(data[i, j], None), **kw)
            texts.append(text)

    return texts

def plot_heatmap(result_dict):
    y_labels = ["Correct", "Not Negation", "Random Deletion", "Random Masking", "Span Deletion", "Word Reordering"]
    x_labels = []
    result_list = []
    for label, result in result_dict.items():
        x_labels.append(label)
        result_list.append(result)

    result_matrix = np.array(result_list).transpose()
    fig, ax = plt.subplots()
    ax.matshow(result_matrix, cmap=plt.cm.Spectral)
    result_string = [["{:.2f}".format(ele) for ele in row] for row in result_matrix]
    ax.set(xticks=np.arange(len(x_labels)), xticklabels=x_labels, yticks=np.arange(len(y_labels)), yticklabels=y_labels)
    # Rotate the tick labels and set their alignment.
    plt.setp(ax.get_xticklabels(), rotation=45, ha="left",
             rotation_mode="anchor")

    # Loop over data dimensions and create text annotations.
    for i in range(result_matrix.shape[0]):
        for j in range(result_matrix.shape[1]):
            text = ax.text(i, j, result_string[i][j],
                           ha="center", va="center", color="w")


    plt.show()

if __name__ == '__main__':
    input_path_list = ["/home/zxj/Data/relation_based_analogy/relation_analogy_per_category_add", "/home/zxj/Data/relation_based_analogy/relation_analogy_per_category_mul"]

    name_list = ['GLOVE', 'DCT(k=1)', 'SkipThought', 'QuickThought', 'InferSentV1', 'InferSentV2', 'GenSen', 'USE-DAN', 'USE-Transformer', 'BERT-BASE-AVG', 'BERT-LARGE-AVG', 'XLNET-BASE-AVG', 'XLNET-LARGE-AVG', 'ROBERTA-BASE-AVG', 'ROBERTA-LARGE-AVG', 'SBERT-BASE-CLS', 'SBERT-LARGE-CLS', 'SRoBERTa-BASE-AVG', 'SRoBERTa-LARGE-AVG']
    y_labels = ["Correct", "Not Negation", "Random Deletion", "Random Masking", "Span Deletion", "Word Reordering"]
    dataset_names = ["Entailment", "Negation", "Passivization", "Objective Clause", "Predicative Adjective Conversion"]
    ALIAS_DICT = {"$c^0$": "DCT(k=0)", "$c^{0:1}$": "DCT(k=1)", "$c^{0:2}$": "DCT(k=2)", "$c^{0:3}$": "DCT(k=3)",
                  "$c^{0:4}$": "DCT(k=4)", "$c^{0:5}$": "DCT(k=5)", "$c^{0:6}$": "DCT(k=6)",
                  "UniversalSentenceDAN": "USE-DAN", "UniversalSentenceTransformer": "USE-Transformer"}
    fig, ax = plt.subplots(2, figsize=(12, 9), dpi=200)
    for idx, input_path in enumerate(input_path_list):
        with open(input_path, "r") as input_file:
            input_dict = json.load(input_file)
            input_dict = {ALIAS_DICT.get(key, key): value for key, value in input_dict.items()}
            input_dict = {key: input_dict[key] for key in name_list}
            result_list = []
            x_labels = []
            for label, result in input_dict.items():
                x_labels.append(label)
                result_list.append(result)
            result_matrix = np.array(result_list).transpose()
            im = heatmap(result_matrix, dataset_names, x_labels, ax=ax[idx],
                         cmap="YlGn")

            texts = annotate_heatmap(im, valfmt="{x:.2f}")

        ''''
        for i, result_dict in enumerate(input_dict.values()):
            result_list = []
            x_labels = []
            for label, result in result_dict.items():
                x_labels.append(label)
                result_list.append(result)
            result_matrix = np.array(result_list).transpose()
            im = heatmap(result_matrix, y_labels, x_labels, ax=ax[i],
                               cmap="YlGn")

            texts = annotate_heatmap(im, valfmt="{x:.2f}")
        '''

        #plt.show()
        plt.savefig("/home/zxj/Data/relation_based_analogy/result/relation_analogy_per_category")
