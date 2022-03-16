import pandas as pd
import numpy as np
from plotly.graph_objs import Bar, Violin

def get_graphs(df):
    '''
    INPUT - (dataframe) data used to generate graphs
    RETURNS - (dict) dictionary to generate plotly JSON graph objects.
    '''

    top5_brands = list(df['brand'].value_counts().head().index.str.title())
    top5_brands_counts = list(df['brand'].value_counts().values)

    graph_one_data = [Bar(
                    x=top5_brands,
                    y=top5_brands_counts,
                    marker={'color':'rgb(220,150,170)'}
                )]

    graph_one_layout = {
                'title': 'Top Five Whisky Brands in Dataset',
                'yaxis': {'title': "Frequency"},
                'xaxis': {'title': "Brand"}
                }
    graphs=[]
    graphs.append(dict(data=graph_one_data, layout=graph_one_layout))
    return graphs
