import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def plot_space(measure):
    plt.style.use(['seaborn-paper','science','ieee','no-latex'])
    matplotlib.rc("font", family="Times New Roman")    
    
    
    data = measure[1]
    g = sns.FacetGrid(
        data,
        col="Parameter",
        col_order=sorted(data["Parameter"].unique()),
        sharex=False)

    # Create the bar plot on each subplot
    g.map(
        sns.barplot,
        "Parameter","value", "variable",
        hue_order=data["variable"].unique())

    
    labels = data["variable"].unique()
    cols = sorted(data["Parameter"].unique())
    axes = np.array(g.axes.flat)
    
    # Iterate over each subplot and set the labels
    for i, ax in enumerate(axes):

        # Set the x-axis ticklabels
        ax.set_xticks([-.25, 0, .25])
        ax.set_xticklabels(labels)

        # Set the label for each subplot
        ax.set_xlabel(cols[i])
        
        # Remove the y-axis label and title
        ax.set_ylabel("")
        ax.set_title("")
    
    # Set the y-axis label only for the left subplot
    axes.flat[0].set_ylabel("Amount")
    
    # Remove the "spines" (the lines surrounding the subplot)
    sns.despine(ax=axes[1], left=True)

    # Set the overall title for the plot
    plt.gcf().suptitle("Stace, Decision and Total Space for Different Parameters", fontsize=12, x=0.55)
    plt.tight_layout()
    
    return plt.gcf()


