import seaborn as sns
import matplotlib.pyplot as plt

# Text wrap for nicer graph output
import textwrap
def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)
    
def plot_choice_per_vehicle_model(df):
    # plot out top 10 vehicle brands at charging station
    sns.histplot(df, hue='vehicle_model', hue_order=df.vehicle_model.value_counts().iloc[:10].index, multiple='dodge')
    
def plot_correlation(df):
    # plot correlation matrix of dataframe
    plt.figure(figsize=(16, 10))
    sns.heatmap(session_data.corr(), annot=True)