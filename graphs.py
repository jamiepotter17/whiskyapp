import pandas as pd
import numpy as np
from plotly.graph_objs import Bar, Scatter3d, Layout
from plotly.graph_objs.layout import Scene
from plotly.graph_objs.layout.scene import XAxis, YAxis, ZAxis

def get_dataset_graphs(df):
    '''
    INPUT - (dataframe) data used to generate graphs
    RETURNS - (dict) dictionary to generate plotly JSON graph objects.
    '''
    regions = list(df['region'].value_counts().index.str.title())
    region_counts = list(df['region'].value_counts().values)

    graph_one_data = [Bar(
                    x=regions,
                    y=region_counts,
                    marker={'color':'rgb(200,150,20)'}
                )]

    graph_one_layout = {
                'title': dict(text ='Regions/Types in Training Data',
                              xanchor='center', x=0.5),
                'yaxis': {'title': "Frequency"},
                'plot_bgcolor': 'rgb(245,245,235)'
                }

    top10_brands = list(df['brand'].value_counts().head(10).index.str.title())
    top10_brands_counts = list(df['brand'].value_counts().head(10).values)

    graph_two_data = [Bar(
                    x=top10_brands,
                    y=top10_brands_counts,
                    marker={'color':'rgb(210,130,90)'}
                )]

    graph_two_layout = {
                'title': dict(text ='Most Reviewed Brands in Training Data',
                              xanchor='center', x=0.5),
                'yaxis': {'title': "Frequency"},
                'plot_bgcolor': 'rgb(245,245,235)'
                }
    graphs=[]
    graphs.append(dict(data=graph_one_data, layout=graph_one_layout))
    graphs.append(dict(data=graph_two_data, layout=graph_two_layout))
    return graphs

def get_distance_graph(distances_df, guess):
    '''
    INPUTS:
    distances_df - dataframe with nose, palate, finish and overall distances
                    computed between each whisky. Uses multi-index.
    guess - string value of the guess returned by the predictive model. Need
            this so that it creates the write 3d scatterplot.
    OUTPUTS:
    graph_data - Plotly Scatter3d object ready to be encoded
    graph_layout - Plotly layout object ready to be encoded
    '''
    graph_data = Scatter3d(
    x=distances_df.loc[(guess,'nose_d')].values,
    y=distances_df.loc[(guess,'palate_d')].values,
    z=distances_df.loc[(guess,'finish_d')].values,
    hoverinfo='text',
    hovertext = distances_df.loc[(guess,'nose_d')].index,
    mode='markers+text',
    marker=dict(size=10,
                color=distances_df.loc[(guess,'overall_d')].values,
                colorscale='sunset',
                opacity=0.7))

    graph_layout = Layout(
    title = dict(text='Whiskies Similar to '+guess, xanchor='center', x=0.5),
    scene = Scene(
        xaxis=XAxis(title='Nose', showticklabels=False), yaxis=YAxis(title='Palate', showticklabels=False),
        zaxis=ZAxis(title='Finish', showticklabels=False),
        camera = dict(eye = dict(x=1.2, y=-1.2, z=1.2))))

    return graph_data, graph_layout
