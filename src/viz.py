# Text wrap for nicer graph output
import textwrap
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
import plotly
import plotly.graph_objects as go
import plotly.io as pio
from plotly.subplots import make_subplots
from mpl_toolkits.mplot3d import Axes3D
from pylab import *

def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)

def energyDdelivered(df):
    """
    Visualize the relations between delivered energy (column name: cumEnergy_Wh) and several other key metrics
    df: Pandas Dataframe
    """
    pass


def draw_barchart(x, y):
    pass


def draw_boxplot(x, y):
    """
    Draw a box plot.
    x: Values
    y: Labels
    """
    pass





# df1 = pd.read_csv("../data/Sessions2_20221020.csv")
# energyDdelivered(df1)

