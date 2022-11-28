# Text wrap for nicer graph output
import textwrap
from process import unpack_power


def wrap_labels(ax, width, break_long_words=False):
    labels = []
    for label in ax.get_xticklabels():
        text = label.get_text()
        labels.append(textwrap.fill(text, width=width,
                      break_long_words=break_long_words))
    ax.set_xticklabels(labels, rotation=0)


def viz_charging_profile(power_row):
    """Plots power profile for a given charging session

    Parameters
    ----------
    power_row :
        row containing power data in string format

    Returns
    -------
    ax : matplotlib axis
        plot of the charging profile

    """
    power_df = unpack_power(power_row['power'])
    power_plot = power_df.plot(x='timestamp', y='power_kw')
    return power_plot
