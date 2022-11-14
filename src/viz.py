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


def plot_choice_per_vehicle_model(df, vehicle_models=['Model 3', 'Prius Prime', 'Volt', 'Bolt EV']):
    # plot vehicle brand count at charging station per vehicle model
    plt.figure(figsize=(20, 10))
    sns.histplot(df[df.vehicle_model.isin(vehicle_models)], x='choice', hue='vehicle_model', multiple='dodge')


def plot_choice_per_site(df):
    # plot out choice selection count per site
    plt.figure(figsize=(20, 10))
    sns.histplot(df, x='choice', hue='siteId', multiple='dodge')


def plot_cumulative_energy(df):
    # plot cumulative energy histogram per choice
    plt.figure(figsize=(20, 10))
    sns.histplot(df, x='cumEnergy_Wh', hue='choice', multiple='dodge')


def plot_correlation(df):
    # plot correlation matrix of dataframe
    plt.figure(figsize=(16, 10))
    sns.heatmap(df.corr(), annot=True)