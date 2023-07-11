import pandas as pd
from matplotlib import rcParams
import plotly.graph_objects as go

rcParams['figure.figsize'] = 10,5

def get_line_colors():

    return ['darkblue',
            'deeppink',
            'teal',
            'dimgrey',
            'black',
            'green',
            'purple']

def plot_loss(data):

    line_colors = get_line_colors()

    if not isinstance(data, pd.DataFrame):
        data = pd.DataFrame(data, columns=["loss"])

    fig = go.Figure()

    for col_idx, col in enumerate(data.columns):

        fig.add_trace(go.Scatter(
            x=data.index+1,
            y=data[col],
            name=col,
            line_color=line_colors[col_idx],
            opacity=.75))

    fig.update_yaxes(type="log")

    fig.show()

