import re

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder

antibiotics = ["VANCO"]
treatments = ["DONOR", "RECIPIENT"]

light_blue = plt.rcParams['axes.prop_cycle'].by_key()['color'][0]
orange = plt.rcParams['axes.prop_cycle'].by_key()['color'][1]


def merge_data_yasmin(normalize=True):
    df = pd.DataFrame()
    for abx in antibiotics + ["PBS"]:
        for treatment in treatments:
            temp = pd.read_csv(f"../Data/Yasmin_FMT/{abx}_{treatment}_genes_norm_named.tsv", sep="\t")
            temp = temp.drop("gene_id", axis=1).set_index("gene_name")
            # sum all rows with the same gene_name
            temp = temp.groupby(temp.index).sum()
            # normalize the data so column will sum to 1_000_000
            print(abx, treatment)
            sum_reads = temp.sum(axis=0)
            # print the sum of reads for each sample in scientific notation
            print(f"sum of reads for each sample: {sum_reads.mean():.1E}", f"+-{sum_reads.std():.1E}")
            if normalize:
                temp = temp.div(temp.sum(axis=0), axis=1) * 1_000_000
            # merge temp with df based on index
            df = pd.concat([df, temp], axis=1)
    # save df to a file
    df.to_csv(f"./Private/Yasmin_FMT_merged{'_normalized' if normalize else ''}.tsv", sep="\t",
              index=True)
    return df


def four_way_forest(df, feature_columns, target_column, test_size=8 / 28, random_state=42):
    """
    Perform classification with a random forest classifier for four classes.

    Parameters:
    - df: DataFrame with features and target variable.
    - feature_columns: List of column names for features.
    - target_column: Name of the target variable column.
    - test_size: Proportion of the data to include in the test split (default is 0.2).
    - random_state: Seed for random number generation (default is 42).

    Returns:
    - clf: Trained random forest classifier.
    - conf_matrix: Confusion matrix.
    - classification_rep: Classification report.
    """
    # Split the data into features (X) and target variable (y)
    X = df[feature_columns]
    y = df[target_column]
    # Encode target variable if it's not numeric
    label_encoder = LabelEncoder()
    y = label_encoder.fit_transform(y)
    # Split the data into training and testing sets
    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=random_state,
                                                        stratify=y)
    # Build a random forest classifier
    clf = RandomForestClassifier(random_state=random_state)
    clf.fit(X_train, y_train)
    # Predictions on the test set
    y_pred = clf.predict(X_test)
    # Get actual labels before encoding
    actual_labels = label_encoder.inverse_transform(y_test)
    # Evaluate the classifier
    conf_matrix = confusion_matrix(y_test, y_pred)
    classification_rep = classification_report(y_test, y_pred)
    # create a dictionary of the labels using the label encoder
    labels_dict = {i: label for i, label in enumerate(label_encoder.classes_)}

    print("Actual Labels:")
    print(labels_dict)
    print("\nConfusion Matrix:")
    print(conf_matrix)
    print("\nClassification Report:")
    print(classification_rep)
    # convert classification report to a DataFrame
    report = classification_report_to_df(classification_rep)
    # get features importance
    importance = pd.Series(clf.feature_importances_, index=feature_columns)

    return conf_matrix, report.values, importance


def classification_report_to_df(report_str):
    lines = report_str.split('\n')
    # Find the column names
    column_names = re.findall(r'\b\w+\b', lines[0])
    # Initialize an empty list to store data
    data = []
    # Iterate through the lines and extract data
    for line in lines[2:-5]:
        values = re.findall(r'\b\d+\.?\d*\b', line)
        if values:
            data.append(values)
    # Create a Pandas DataFrame
    df = pd.DataFrame(data, columns=column_names)
    # set precision to index, rename recall to precision, f1 to recall, score to f1-score
    df = df.rename(columns={"precision": "index", "recall": "precision", "f1": "recall", "score": "f1-score"})
    df = df.set_index("index")
    # convert all values to float
    df = df.astype(float)

    return df


def four_way_random_forest(data, metadata, reps=10000):
    # add to data the gropu column from metadata
    data = data.T
    data = pd.merge(data, metadata[["ID", "group"]], left_index=True, right_on="ID").set_index("ID")
    confusion_matrix = np.zeros((4, 4))
    classification_report = np.zeros((4, 4))
    importance = pd.Series(np.zeros(len(data.columns[:-1])), index=data.columns[:-1])
    for i in range(reps):
        result = four_way_forest(data, data.columns[:-1], "group", test_size=8 / 28, random_state=i)
        confusion_matrix += result[0]
        classification_report += result[1]
        importance += result[2]
    confusion_matrix /= reps
    classification_report /= reps
    importance /= reps
    print("Confusion Matrix:")
    print(confusion_matrix)
    # make a df from the confusion matrix with columns |  PBS_DONOR   |  PBS_RECIPIENT  |  Vanco_DONOR  | Vanco_RECIPIENT |
    # and rows |  PBS_DONOR   |  PBS_RECIPIENT  |  Vanco_DONOR  | Vanco_RECIPIENT |
    confusion_matrix = pd.DataFrame(confusion_matrix,
                                    columns=["PBS Donor", "PBS Recipient", "Van Donor", "Van Recipient"],
                                    index=["PBS Donor", "PBS Recipient", "Van Donor", "Van Recipient"])
    # save the confusion matrix
    confusion_matrix.to_csv("./Private/confusion_matrix.csv", index=True)
    classification_report = pd.DataFrame(classification_report, columns=["precision", "recall", "f1-score", "support"])
    print("\nClassification Report:")
    print(classification_report)
    # save feature importance
    importance = importance.sort_values(ascending=False)
    importance.to_csv("./Private/feature_importance.csv", index=True)
    return confusion_matrix


def analyze_genes_results(data, metadata, sizes=(100, 200, 400), abx=None, random=False):
    # change group column to group (ID)
    # metadata["group"] = metadata["group"] + "(" + metadata["ID"] + ")"
    metadata["Drug"] = metadata["Drug"].replace("VANCO", "Van")
    if not abx:
        metadata["Treatment"] = metadata["Treatment"].str.capitalize()
    metadata["group"] = metadata["Treatment"] + " " + metadata["Drug"]

    feature_importance = pd.read_csv("./Private/feature_importance.csv", index_col=0)
    # rename index to "gene" and rename first column to "importance"
    feature_importance.index.name = "gene"
    feature_importance = feature_importance.rename(columns={feature_importance.columns[0]: "importance"})
    for num_top in sizes:
        for col_cluster in [False, True]:
            top = feature_importance.head(num_top).index.values
            if abx:
                top = [gene for gene in top if gene in data.index]
                if random:
                    top = np.random.choice(data.index, 100, replace=False)

                pbs_columns = [col for col in data.columns if "C" in col]
                non_pbs_columns = [col for col in data.columns if "C" not in col]
                ordered_columns = pbs_columns + non_pbs_columns
                data = data[ordered_columns]
            # create a df with only those genes
            top_df = data.loc[top]

            # rename the columns to the group
            top_df = top_df.rename(columns=metadata.set_index('ID')['group'].to_dict())

            # cluster the genes using hierarchical clustering
            cluster = sns.clustermap(top_df, row_cluster=True, metric="euclidean", method="average",
                                     col_cluster=col_cluster, z_score=0)
            # increase x labels font size
            plt.xticks(fontsize=16)
            # get the order of the genes after clustering
            # genes = cluster.data2d.index
            genes = cluster.dendrogram_row.reordered_ind
            # show all y labels
            plt.yticks(np.arange(len(top_df.index)), top_df.index, rotation=0, fontsize=8)
            # remove y label
            plt.ylabel("")
            # increase figure size
            plt.gcf().set_size_inches(15, 15 * num_top / 100)
            # save the clustermap
            plt.savefig(
                f"./Private/feature_importance_{num_top}_{'col_clustered_' if col_cluster else ''}clustermap{'_' + abx if abx else ''}.png")
            plt.close()
            top_df = top_df.iloc[genes]
            top_df.to_csv(
                f"./Private/feature_importance_{num_top}_{'col_clustered_' if col_cluster else ''}clustered{'_' + abx if abx else ''}.csv")
            if not abx:
                plot_heatmap_colors(cluster, col_cluster,
                                    f"feature_importance_{num_top}_{'col_clustered_' if col_cluster else ''}heatmap{'_' + abx if abx else ''}",
                                    top_df)
            else:
                colors = {'IP PBS': 'blue', 'IV PBS': light_blue, f'IP {abx.split("_")[0].capitalize()}': 'red',
                          f'IV {abx.split("_")[0].capitalize()}': orange, 'PO PBS': 'navy',
                          f'PO {abx.split("_")[0].capitalize()}': 'yellow'}
                plot_heatmap_colors(cluster, col_cluster,
                                    f"feature_importance_{num_top}_{'col_clustered_' if col_cluster else ''}heatmap{'_' + abx if abx else ''}",
                                    top_df, jump=6, xticks=[3, 8.5], colors=colors, normalize=False, sort=False)
                # top_df, jump=6, xticks=[2.5, 7.5, 13, 18.5, 24, 30], colors=colors, normalize=False)


def plot_heatmap_colors(cluster, col_cluster, save_name, top_df, normalize=True, colors=None, title="", jump=7,
                        xticks=[3.5, 10.5, 17.5, 24.5], show_all_y=True, hline=None, vline=False, sort=True):
    if col_cluster:
        mice = cluster.dendrogram_col.reordered_ind
        top_df = top_df.iloc[:, mice]
    if normalize:
        # remove from each row its mean
        top_df = top_df.sub(top_df.mean(axis=1), axis=0)
        # normalize each row by its standard deviation
        top_df = top_df.div(top_df.std(axis=1), axis=0)
    if sort:
        # sort top_df columns lexically
        columns = np.argsort(top_df.columns)
        top_df = top_df.iloc[:, columns]

    # plot the heatmap
    vmax = 6 if show_all_y else 5
    if not show_all_y:
        heatmap = sns.heatmap(top_df, vmax=vmax, cmap="RdBu_r")
    else:
        # heatmap = sns.heatmap(top_df)
        heatmap = sns.heatmap(top_df, vmax=vmax, cmap="RdBu_r")
    # create a dictionary 'PBS DONOR': 'blue', 'PBS RECIPIENT': 'light blue',
    # 'Van DONOR': 'red', 'Van RECIPIENT': 'light red'
    cbar = heatmap.collections[0].colorbar
    actual_vmin, actual_vmax = cbar.vmin, cbar.vmax
    # increase tick size
    cbar.ax.tick_params(labelsize=15)
    if actual_vmax == vmax:
        # Define explicit ticks, ensuring they cover your desired range
        ticks = np.linspace(actual_vmin, actual_vmax, num=5)  # Adjust the number of ticks as needed
        # Set the ticks on the colorbar
        cbar.set_ticks(ticks)
        # Customize tick labels, modifying the last label to indicate a limit
        last = f"{ticks[-1]:.0f}+" if actual_vmax == vmax else actual_vmax
        tick_labels = [f"{tick:.0f}" for tick in ticks[:-1]] + [last]  # Add '+' to the last label
        # Apply the customized tick labels
        cbar.set_ticklabels(tick_labels)

    if not colors:
        colors = {'Donor PBS': 'blue', 'Recipient PBS': light_blue, 'Donor Van': 'red', 'Recipient Van': orange}
    label_colors = [colors[label] for label in top_df.columns]
    if hline:
        plt.axhline(y=hline, color='black', linewidth=1)
    if vline:
        for i in range(1, 3):
            # draw a line between pbs and abx
            plt.axvline(x=14 * i - 7, color="black", linewidth=1, linestyle="--")
            # draw a dashed line between time points
            plt.axvline(x=14 * i, color="black", linewidth=1)
    bar_height = 0.01 * top_df.shape[0]
    ax = plt.gca()
    for k, color in enumerate(label_colors):
        bar_width = 1  # Set the width to match a column (fixed at 1)
        ax.add_patch(
            plt.Rectangle((k, top_df.shape[0] - bar_height), bar_width, bar_height, color=color,
                          fill=True))
    # show only the 4th, 11th, 18th, 25th columns
    ax.set_xticks(xticks)
    ax.set_xticklabels(top_df.columns[::jump])
    plt.xticks(rotation=45, ha='center', size=20)
    if title:
        plt.title(title)
    # plt.xticks(np.arange(len(top_df.columns)), top_df.columns, rotation=45, ha='right', fontsize=12)
    if show_all_y:
        # show all yticks
        plt.yticks(np.arange(len(top_df.index)), top_df.index, rotation=0, fontsize=8)
    plt.ylabel("")
    # increase all font sizes
    plt.rc('font', size=25)
    plt.gcf().set_size_inches(2.5 * (len(colors) + 1), 15 * top_df.shape[0] / 100)
    plt.tight_layout()
    plt.savefig(f"./Private/{save_name}.pdf", dpi=600)
    plt.close()


def plot_heatmap_colors_short(save_name, top_df, normalize=True, colors=None, title="", jump=7,
                              xticks=[3.5, 10.5, 17.5, 24.5], hline=None, vline=False, sort=True):
    if normalize:
        # remove from each row its mean
        top_df = top_df.sub(top_df.mean(axis=1), axis=0)
        # normalize each row by its standard deviation
        top_df = top_df.div(top_df.std(axis=1), axis=0)
    if sort:
        # sort top_df columns lexically
        columns = np.argsort(top_df.columns)
        top_df = top_df.iloc[:, columns]

    sns.heatmap(top_df, cmap="RdBu_r")
    if not colors:
        colors = {'Donor PBS': 'blue', 'Recipient PBS': light_blue, 'Donor Van': 'red', 'Recipient Van': orange}
    label_colors = [colors[label] for label in top_df.columns]
    if hline:
        plt.axhline(y=hline, color='black', linewidth=1)
    if vline:
        for i in range(1, 3):
            # draw a line between pbs and abx
            plt.axvline(x=14 * i - 7, color="black", linewidth=1, linestyle="--")
            # draw a dashed line between time points
            plt.axvline(x=14 * i, color="black", linewidth=1)
    bar_height = 0.01 * top_df.shape[0]
    ax = plt.gca()
    for k, color in enumerate(label_colors):
        bar_width = 1  # Set the width to match a column (fixed at 1)
        ax.add_patch(
            plt.Rectangle((k, top_df.shape[0] - bar_height), bar_width, bar_height, color=color,
                          fill=True))
    # show only the 4th, 11th, 18th, 25th columns
    ax.set_xticks(xticks)
    ax.set_xticklabels(top_df.columns[::jump])
    plt.xticks(rotation=45, ha='center', size=20)
    plt.yticks(np.arange(len(top_df.index)), top_df.index, rotation=0, fontsize=8)
    if title:
        plt.title(title)
    # plt.xticks(np.arange(len(top_df.columns)), top_df.columns, rotation=45, ha='right', fontsize=12)
    plt.ylabel("")
    # increase all font sizes
    plt.rc('font', size=25)
    plt.gcf().set_size_inches(2.5 * (len(colors) + 1), 15 * top_df.shape[0] / 100)
    if save_name == "down_regulated":
        plt.gcf().set_size_inches(2.5 * (len(colors) + 1), 4 * 15 * top_df.shape[0] / 100)
    plt.tight_layout()
    # plt.savefig(f"./Private/YasminRandomForest/{save_name}.png", dpi=600)
    plt.savefig(f"./Private/{save_name}.pdf", dpi=600)
    plt.close()


def calc_all_statistics(df, meta):
    # remove empty rows
    df = df.loc[~(df == 0).all(axis=1)]
    from scipy.stats import ttest_ind
    all_stats = pd.DataFrame()
    all_stats.index = df.index
    for treat in treatments:
        abx_data = meta[(meta['Drug'] == "VANCO") & (meta["Treatment"] == treat)]['ID'].values
        pbs_data = meta[(meta['Drug'] == 'PBS') & (meta["Treatment"] == treat)]['ID'].values
        # temp = raw.loc[np.concatenate(abx_data, pbs_data)]
        ttest_pvalues = df.apply(lambda row: ttest_ind(row[abx_data], row[pbs_data])[1], axis=1)
        # Calculate the fold changes
        fold_changes = df.apply(lambda row: np.log2(np.median(row[abx_data]) / np.median(row[pbs_data])), axis=1)
        # add the p-values and fold changes to the all_stats df
        all_stats = pd.concat([all_stats, pd.DataFrame(
            {f"p-value_{treat}": ttest_pvalues, f"fold_change_{treat}": fold_changes}, index=df.index)], axis=1)
    for abx in antibiotics + ["PBS"]:
        donor_data = meta[(meta['Drug'] == abx) & (meta["Treatment"] == "DONOR")]['ID'].values
        recipient_data = meta[(meta['Drug'] == abx) & (meta["Treatment"] == "RECIPIENT")]['ID'].values
        # temp = raw.loc[np.concatenate(abx_data, pbs_data)]
        ttest_pvalues = df.apply(lambda row: ttest_ind(row[donor_data], row[recipient_data])[1], axis=1)
        # Calculate the fold changes
        fold_changes = df.apply(lambda row: np.log2(np.median(row[donor_data]) / np.median(row[recipient_data])),
                                axis=1)
        # add the p-values and fold changes to the all_stats df
        all_stats = pd.concat([all_stats, pd.DataFrame(
            {f"p-value_{abx}": ttest_pvalues, f"fold_change_{abx}": fold_changes}, index=df.index)], axis=1)

    donor_van = meta[(meta['Drug'] == "VANCO") & (meta["Treatment"] == "DONOR")]['ID'].values
    recipient_pbs = meta[(meta['Drug'] == "PBS") & (meta["Treatment"] == "RECIPIENT")]['ID'].values
    # temp = raw.loc[np.concatenate(abx_data, pbs_data)]
    ttest_pvalues = df.apply(lambda row: ttest_ind(row[donor_van], row[recipient_pbs])[1], axis=1)
    # Calculate the fold changes
    fold_changes = df.apply(lambda row: np.log2(np.median(row[donor_van]) / np.median(row[recipient_pbs])),
                            axis=1)
    # add the p-values and fold changes to the all_stats df
    all_stats = pd.concat([all_stats, pd.DataFrame(
        {f"p-value_van_donor_pbs_recipient": ttest_pvalues, f"fold_change_van_donor_pbs_recipient": fold_changes},
        index=df.index)], axis=1)

    donor_pbs = meta[(meta['Drug'] == "PBS") & (meta["Treatment"] == "DONOR")]['ID'].values
    recipient_van = meta[(meta['Drug'] == "VANCO") & (meta["Treatment"] == "RECIPIENT")]['ID'].values
    # temp = raw.loc[np.concatenate(abx_data, pbs_data)]
    ttest_pvalues = df.apply(lambda row: ttest_ind(row[donor_pbs], row[recipient_van])[1], axis=1)
    # Calculate the fold changes
    fold_changes = df.apply(lambda row: np.log2(np.median(row[donor_pbs]) / np.median(row[recipient_van])),
                            axis=1)
    # add the p-values and fold changes to the all_stats df
    all_stats = pd.concat([all_stats, pd.DataFrame(
        {f"p-value_pbs_donor_van_recipient": ttest_pvalues, f"fold_change_pbs_donor_van_recipient": fold_changes},
        index=df.index)], axis=1)

    all_stats.to_csv("./Private/all_stats.csv")


def plot_confusion_matrix():
    # plot it as a heatmap, make x label "predicted", y label "true"
    import matplotlib.pyplot as plt
    import seaborn as sns
    forest_confusion_matrix = pd.read_csv("./Private/confusion_matrix.csv", index_col=0)
    fig, ax = plt.subplots(figsize=(10, 10))
    # increase size of number on the heatmap
    sns.heatmap(forest_confusion_matrix / 2, annot=True, fmt=".2%", ax=ax, annot_kws={"size": 20})
    ax.tick_params(labelsize=18)
    ax.set_xlabel("Predicted Category", fontsize=25)
    ax.set_ylabel("True Category", fontsize=25)
    plt.tight_layout()
    plt.savefig("./Private/confusion_matrix.png")
    plt.close()


def neo_van_ip_significant_same_trend():
    all_stat = pd.read_csv("./Private/all_stats.csv")
    # keep only rows where p-value_Neo_IP < 0.05 and p-value_Van_IP < 0.05
    all_stat = all_stat[(all_stat["p-value_Neo_IP"] < 0.05) & (all_stat["p-value_Van_IP"] < 0.05)]
    # keep only rows where fold_change_Neo_IP and fold_change_Van_IP has the same sign
    all_stat = all_stat[(all_stat["fold_change_Neo_IP"] * all_stat["fold_change_Van_IP"]) > 0]
    # save all_stat as a csv file named neo_van_ip_significant.csv
    all_stat.to_csv("./Private/neo_van_ip_significant.csv", index=False)


def plot_neo_van_clusters():
    # read the metadata
    meta = pd.read_csv("./Private/neo_van_ip_metadata.csv")
    # read the transcriptome
    transcriptome = pd.read_csv("./Private/neo_van_ip.csv", index_col=0)
    genes = pd.read_csv(f"./Private/neo_van_ip_significant.csv")
    transcriptome = transcriptome.loc[genes["gene_name"]]

    # plot the heatmap
    colors = {'PBS': 'blue', 'Neo': 'red', 'Van': orange}

    pbs = meta[meta["Drug"] == "PBS"]
    transcriptome.to_csv("./Private/neo_van_ip_significant_genes.csv")
    # z-score trancriptome by pbs
    transcriptome = transcriptome.sub(transcriptome[pbs["ID"]].mean(axis=1), axis=0)
    transcriptome = transcriptome.div(transcriptome[pbs["ID"]].std(axis=1), axis=0)
    transcriptome.to_csv("./Private/neo_van_ip_significant_genes_zscore.csv")

    cluster = sns.clustermap(transcriptome, metric="correlation", method="average", row_cluster=True,
                             col_cluster=False)
    genes = cluster.dendrogram_row.reordered_ind
    transcriptome = transcriptome.iloc[genes]
    plt.close()

    names_dict = meta.set_index('ID')['Drug'].to_dict()
    transcriptome = transcriptome.rename(columns=names_dict)

    plot_heatmap_colors(None, False, "neo_van_ip_heatmap", transcriptome, normalize=False,
                        colors=colors, title="Neo vs. Van IP significant genes", jump=6, xticks=[3, 8.5, 13.5],
                        show_all_y=False)


if __name__ == "__main__":
    metadata = pd.read_csv("../Data/Yasmin_FMT/metadata-Yasmin_FMT.tsv", sep="\t")
    # create columns Drug and Treatment, based on group.split("_") accordingly ([0] is drug, [1] is treatment)
    metadata["Drug"] = metadata["group"].apply(lambda x: x.split("_")[0])
    metadata["Treatment"] = metadata["group"].apply(lambda x: x.split("_")[1])
    metadata = metadata.rename(columns={"sample": "ID"})
    data = merge_data_yasmin()

    # four_way_random_forest(data, metadata)
    # plot_confusion_matrix()
    # analyze_genes_results(data, metadata, sizes=(50, 100, 200, 400, 500))

    # calc_all_statistics(data, metadata)
    neo_van_ip_significant_same_trend()
    plot_neo_van_clusters()
