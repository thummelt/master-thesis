import seaborn as sns 
import matplotlib
import matplotlib.pyplot as plt
import numpy as np

def save(plts, transparent: bool = False):
    # Export plots
    for pl in plts:
        pl[1].savefig('/usr/app/output/graphics/%s.svg' % pl[0], transparent=transparent, bbox_inches='tight')
        pl[1].savefig('/usr/app/output/graphics/%s.png' % pl[0], transparent=transparent, bbox_inches='tight')


def formatPlot(g, xlabel, ylabel, title,xticks=None,dense=None,legend_t = None, legend_opt = None, legend_loc = "upper_left",  yticks = None ):
    if xticks is not None:
        g.set(xticks=xticks)

    if dense is not None:
        for i, ax in enumerate(np.array(g.axes.flat)):
            labels = list(ax.get_xticklabels())
            ax.set_xticklabels([b if i%dense == 0 else "" for i,b in enumerate(labels)], rotation=40, ha="right")
        
    
    if yticks is not None:
        g.set(yticks = yticks)

    plt.xlabel(xlabel)
    plt.ylabel(ylabel)
    plt.title(title)

    if legend_t is not None:
        print(2)
        plt.legend(title=legend_t, loc=legend_loc, labels=legend_opt)



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


