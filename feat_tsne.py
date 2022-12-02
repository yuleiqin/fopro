import numpy as np
from sklearn.manifold import TSNE
import os
import matplotlib.pyplot as plt
import matplotlib
from tqdm import tqdm


MOUSE_10X_COLORS = {
    0: "#00846F",
    1: "#1CE6FF",
    2: "#FF34FF",
    3: "#FF4A46",
    4: "#008941",
    5: "#006FA6",
    6: "#A30059",
    7: "#FFDBE5",
    8: "#7A4900",
    9: "#0000A6",
    10: "#63FFAC",
    11: "#B79762",
    12: "#004D43",
    13: "#8FB0FF",
    14: "#997D87",
    15: "#5A0007",
    16: "#809693",
    17: "#FEFFE6",
    18: "#1B4400",
    19: "#4FC601",
    20: "#3B5DFF",
    21: "#4A3B53",
    22: "#FF2F80",
    23: "#61615A",
    24: "#BA0900",
    25: "#6B7900",
    26: "#00C2A0",
    27: "#FFAA92",
    28: "#FF90C9",
    29: "#B903AA",
    30: "#D16100",
    31: "#DDEFFF",
    32: "#000035",
    33: "#7B4F4B",
    34: "#A1C299",
    35: "#300018",
    36: "#0AA6D8",
    37: "#013349",
    38: "#FFFF00",
}


def plot(x, y, domain, save_path_fig, class_id2name, ax=None, title=None, draw_legend=True,\
    draw_centers=False, draw_cluster_labels=False, colors=None,\
        legend_kwargs=None, label_order=None, **kwargs):
    if ax is None:
        ## figsize=(8, 8)
        _, ax = plt.subplots()
    if title is not None:
        ax.set_title(title)
    plot_params = {"alpha": kwargs.get("alpha", 0.1), "s": kwargs.get("s", 12)}
    # plt.subplots_adjust(right=0.6)
    ## Create main plot
    if label_order is not None:
        assert all(np.isin(np.unique(y), label_order))
        classes = [l for l in label_order if l in np.unique(y)]
    else:
        classes = np.unique(y)
    if colors is None:
        default_colors = matplotlib.rcParams["axes.prop_cycle"]
        colors = {k: v["color"] for k, v in zip(classes, default_colors())}
    point_colors = list(map(colors.get, y))
    ax.scatter(x[:,0], x[:,1], c=point_colors, \
        rasterized=True, **plot_params)
    # Plot mediods
    if draw_centers:
        centers = []
        for yi in classes:
            mask = yi == y
            centers.append(np.median(x[mask, :2], axis=0))
        centers = np.array(centers)
        center_colors = list(map(colors.get, classes))
        ax.scatter(
            centers[:, 0], centers[:, 1], c=center_colors, s=48, alpha=1, edgecolor="k"
        )
        # Draw mediod labels
        if draw_cluster_labels:
            for idx, label in enumerate(classes):
                ## label_text = label
                label_text = class_id2name[int(label)]
                ax.text(centers[idx, 0], centers[idx, 1] + 2.2,\
                    label_text, fontsize=kwargs.get("fontsize", 6),\
                        horizontalalignment="center")
    # Plot prototypes
    prototypes = []
    for yi in classes:
        mask = (yi == y)
        ## domain == 0代表web images
        ## domain == 1代表fewshot samples
        ## domain == 2代表prototypes
        mask = mask & (domain == 2)
        prototypes.append(x[mask, :2])
    prototypes = np.concatenate(prototypes, axis=0)
    center_colors = list(map(colors.get, classes))
    ax.scatter(
        prototypes[:, 0], prototypes[:, 1],\
            marker="X", c=center_colors, s=24, alpha=1,
    )

    # Plot fewShot samples
    mask = (domain == 1)
    fewshots = x[mask]
    fewshots_label = y[mask]

    fewshots_avg = []
    fewshots_label_avg = []
    print("start averaging")
    for label_i in range(int(np.amax(fewshots_label))+1):
        mask_label_i = (fewshots_label == label_i)
        if len(mask_label_i) > 0:
            coord_avg = fewshots[mask_label_i]
            fewshots_avg.append(np.mean(coord_avg, axis=0))
            fewshots_label_avg.append(label_i)
    fewshots = np.array(fewshots_avg)
    fewshots_label = np.array(fewshots_label_avg)

    print("total number of fewshot label = {}".format(fewshots_label.shape[0]))
    center_colors = list(map(colors.get, fewshots_label))
    ax.scatter(
        fewshots[:, 0], fewshots[:, 1],\
            marker="^", c=center_colors, s=24, alpha=1,
    )
    
    # Draw mediod labels
    if draw_cluster_labels:
        for idx, label in enumerate(classes):
            ## label_text = label
            label_text = class_id2name[int(label)]
            ax.text(centers[idx, 0], centers[idx, 1] + 2.2,\
                label_text, fontsize=kwargs.get("fontsize", 6),\
                    horizontalalignment="center")

    # Hide ticks and axis
    ax.set_xticks([]), ax.set_yticks([]), ax.axis("off")
    if draw_legend:
        legend_handles = []
        for yi in classes:
            ## label_text = yi
            label_text = class_id2name[int(yi)]
            legend_handles.append(
                matplotlib.lines.Line2D([], [],\
                    marker="s", color="w", markerfacecolor=colors[yi],\
                        ms=10, alpha=1, linewidth=0, label=label_text, markeredgecolor="k") 
            )
        legend_kwargs_ = dict(loc="center left", bbox_to_anchor=(1, 0.5), frameon=False, )
        if legend_kwargs is not None:
            legend_kwargs_.update(legend_kwargs)
        ax.legend(handles=legend_handles, **legend_kwargs_)
        # plt.show()
        plt.savefig(save_path_fig, dpi=600, bbox_inches='tight')
    return


def dim_reduction_tsne(all_feat, save_path_tsne):
    print("Start TSNE mapping")
    tsne = TSNE(n_components=2).fit_transform(all_feat)
    np.save(save_path_tsne, tsne)
    return tsne


def run_and_plot_tsne(save_root_path, if_average=False):
    """run tsne programme and plot it"""
    feature_npy_path = os.path.join(save_root_path, "WebTrain_feat.npy")
    prototypes_npy_path = os.path.join(save_root_path, "prototypes.npy")
    class_id2name_path = os.path.join(save_root_path, "class_id2name.txt")

    class_id2name = {}
    with open(class_id2name_path, "r") as f:
        lines = f.readlines()
        for line in tqdm(lines):
            info = line.strip().split("@")
            class_id = int(info[0])
            class_name = str(info[1])
            class_id2name[class_id] = class_name
    
    if if_average:
        save_path_tsne = os.path.join(os.path.dirname(feature_npy_path), "tsne_dim_avg5.npy")
        save_path_fig = os.path.join(os.path.dirname(feature_npy_path), "tsne_train_avg5.png")
    else:
        save_path_tsne = os.path.join(os.path.dirname(feature_npy_path), "tsne_dim.npy")
        save_path_fig = os.path.join(os.path.dirname(feature_npy_path), "tsne_train.png")

    features_npy = np.load(feature_npy_path)
    prototypes_npy = np.load(prototypes_npy_path)
    img_domain = features_npy[:,0]
    img_label = features_npy[:,1]
    img_feat = features_npy[:,2:]
    proto_label = np.arange(prototypes_npy.shape[0])
    proto_domain = np.ones(prototypes_npy.shape[0]) * 2
    ## domain 0: web data; 1: fewshot data; 2: prototype
    all_domain = np.concatenate((img_domain, proto_domain), axis=0)
    all_feat = np.concatenate((img_feat, prototypes_npy), axis=0)
    all_label = np.concatenate((img_label, proto_label), axis=0)
    ## 200类过多 只取前20类做渲染
    is_valid = (all_label < 10)
    print(np.sum(is_valid))
    ####8@waffle_iron
    ####12@dam_dike_dyke
    ####18@pole_rod
    ####3@tricycle_trike
    ####13@langur_monkey
    ####9@dining_table
    all_domain = all_domain[is_valid]
    all_feat = all_feat[is_valid]
    all_label = all_label[is_valid]

    # if if_average:
    #     ### Masking all domain of target
    #     print("start averaging")
    #     mask = (all_domain == 1)
    #     fewshots = all_feat[mask]
    #     fewshots_label = all_label[mask]
    #     all_domain = all_domain[~mask]
    #     all_feat = all_feat[~mask]
    #     all_label = all_label[~mask]
    #     fewshots_avg = []
    #     fewshots_label_avg = []
    #     fewshots_domains_avg = []
    #     for label_i in range(int(np.amax(fewshots_label))+1):
    #         mask_label_i = (fewshots_label == label_i)
    #         if len(mask_label_i) > 0:
    #             coord_avg = fewshots[mask_label_i]
    #             fewshots_avg.append(np.mean(coord_avg, axis=0))
    #             fewshots_label_avg.append(label_i)
    #             fewshots_domains_avg.append(1)
    #     fewshots_avg = np.array(fewshots_avg)
    #     fewshots_label_avg = np.array(fewshots_label_avg)
    #     fewshots_domains_avg = np.array(fewshots_domains_avg)
    #     all_domain = np.concatenate((all_domain, fewshots_domains_avg), axis=0)
    #     all_feat = np.concatenate((all_feat, fewshots_avg), axis=0)
    #     all_label = np.concatenate((all_label, fewshots_label_avg), axis=0)

    # if not os.path.exists(save_path_tsne):
    tsne = dim_reduction_tsne(all_feat, save_path_tsne)
    # else:
    #     tsne = np.load(save_path_tsne)
    ##title="t-SNE of web images(o), few-shots(^), and prototypes(x)"

    # is_valid = (all_label == 8) | (all_label == 12) | (all_label == 18) | (all_label == 3) | (all_label == 13) | (all_label == 9)
    # all_domain = all_domain[is_valid]
    # all_feat = all_feat[is_valid]
    # all_label = all_label[is_valid]
    # all_label[all_label==8] = 0
    # all_label[all_label==12] = 1
    # all_label[all_label==18] = 2
    # all_label[all_label==3] = 3
    # all_label[all_label==13] = 4
    # all_label[all_label==9] = 5
    # class_id2name = {}
    # class_id2name[0] = "waffle_iron"
    # class_id2name[1] = "dam_dike_dyke"
    # class_id2name[2] = "pole_rod"
    # class_id2name[3] = "tricycle_trike"
    # class_id2name[4] = "langur_monkey"
    # class_id2name[5] = "dining_table"

    plot(tsne, all_label, all_domain,\
        save_path_fig=save_path_fig,\
            class_id2name=class_id2name,\
                draw_centers=False,\
                    colors=MOUSE_10X_COLORS)
    return

