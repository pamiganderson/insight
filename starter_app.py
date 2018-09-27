# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Show table from database
import time
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
import psycopg2
import pandas as pd
import plotly.graph_objs as go


########## QUERIES ##########
# Define a database name (we're using a dataset on births, so we'll call it birth_db)
# Set your postgres username
dbname = 'fda_adverse_events'
username = 'pami' # change this to your username

### QUERYING THE DATABASE ###
# Connect to make queries using psycopg2
con = None
con = psycopg2.connect(database = dbname, user = username)
# query:
sql_query = """
SELECT *
FROM df_spending_adverse_total;
"""
df = pd.read_sql_query(sql_query,con)

def fetch_data(q):
    result = pd.read_sql(
        sql=q,
        con=con
    )
    return result

def get_generics():
    '''Returns the list of generics that are stored in the database'''

    generic_query = (
        f'''
        SELECT DISTINCT(generic_name)
        FROM df_spending_adverse_total;
        '''
    )
    generics = fetch_data(generic_query)
    generics = list(generics['generic_name'].sort_values(ascending=True))
    return generics

def onLoad_generic_options():
    '''Actions to perform upon initial page load'''

    generic_options = (
        [{'label': generic, 'value': generic}
         for generic in get_generics()]
    )
    return generic_options

# sql_query = """
# SELECT COUNT(Manufacturer)
# FROM df_spending_2012_table
# GROUP BY Generic Name;
# """
# df_spending_2012 = pd.read_sql_query(sql_query,con)


########## FUNCTIONS FOR TABLES AND GRAPHS ##########
def generate_table(dataframe, max_rows=10):
    return html.Table(
        # Header
        [html.Tr([html.Th(col) for col in dataframe.columns])] +

        # Body
        [html.Tr([
            html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
        ]) for i in range(min(len(dataframe), max_rows))]
    )

########## DASH APP ##########
app = dash.Dash()

## CUSTOM COLOR CONFIG ##
colors_banner = {
	'background' : '#ffffff',
	'text': '#000000'
}
colors_light = {
    'background': '#111111',
    'text': '#337cb6'
}
colors_dark = {
    'background': '#111111',
    'text': '#232c63'
}

app.layout = html.Div(children=[
    html.Div([
	    # Page Header
	    html.H1(
	    	children = 'Generic''s For Geriatrics: Drug Assessment',
	    	style = {
	    		'textAlign' : 'center',
	    		'color' : colors_banner['text']
	    	}
		),
	    
	    # Page Sub Header
	    html.H3(
	    	children = 'Patient Tool to Predict Drug Risk',
	    	style = {
	    		'textAlign' : 'center',
	    		'color' : colors_banner['text']
	    	}
		)
	]),

    # Select generic Dropdown
    html.Div([
        html.Div('Select Generic Drug', className='two columns', style={'textAlign' : 'right'}),
        html.Div(dcc.Dropdown(id='generic-selector',
                              options=onLoad_generic_options()),
                 className='one columns',
	    		 style={"width" : "50%"})
    ], className="row"),

    # Page Sub Header
	html.Div([
		html.H3(id='output-risk', style={'textAlign' : 'center', 'color': 'red'}),
	], className="row"),

    # Display the plots for the selected drug
    html.Div([
        html.Div([
            html.H3('# Patients on Drug'),
            dcc.Graph(id='patients_on_drug')
        ], className="six columns"),

        html.Div([
            html.H3('Price Comparator'),
            dcc.Graph(id='price_compare')
        ], className="six columns"),
    ], className="row"),

    # html.Div([
    #     html.Div([
    #         html.H3(''),
    #         dcc.Graph(id='')
    #     ], className="six columns"),

    #     html.Div([
    #         html.H3('Graph 4'),
    #         dcc.Graph(id='g3', figure={'data': [{'y': [1, 2, 3]}]})
    #     ], className="six columns"),
    # ], className="row")

])

# app.css.append_css({
#     'external_url': 'https://codepen.io/chriddyp/pen/bWLwgP.css'
# })

# app.layout = html.Div(children=[
#     html.H4(children='US Agriculture Exports (2011)'),
#     generate_table(df_ad_data_from_sql_q1)
# ])
# Display the adverse events plots
@app.callback(
    dash.dependencies.Output('output-risk', 'children'),
    [dash.dependencies.Input('generic-selector', 'value')])
def model_risk_value(value):
    message = list()
    if value is not None:
    	return '{}'.format(value)

# Display the price difference
@app.callback(
    dash.dependencies.Output('price_compare', 'figure'),
    [dash.dependencies.Input('generic-selector', 'value')])
def update_output(value):
    filtered_df = df[df['generic_name'] == value]
    price_min_max = filtered_df['min_price_per_dose']
    return {
        'data': [go.Bar(
            x=filtered_df['brand_name'],
            y=filtered_df['max_price_per_dose'],
            opacity=0.7,
        )],
        'layout': go.Layout(
            xaxis={'title': 'Drug Name'},
            yaxis={'title': 'Price Per Dosage ($)'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            #legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }

# Display the number of patients on drug difference
@app.callback(
    dash.dependencies.Output('patients_on_drug', 'figure'),
    [dash.dependencies.Input('generic-selector', 'value')])
def update_output(value):
    filtered_df = df[df['generic_name'] == value]
    return {
        'data': [go.Bar(
            x=filtered_df['brand_name'],
            y=filtered_df['total_beneficiaries'],
            opacity=0.7,
        )],
        'layout': go.Layout(
            xaxis={'title': 'Drug Name'},
            yaxis={'title': '# Patients'},
            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
            #legend={'x': 0, 'y': 1},
            hovermode='closest'
        )
    }
# def update_figure(selected_generic):
#     filtered_df = df[df.drug_generic_name == selected_generic]
#     traces = []
#     for i in filtered_df.continent.unique():
#         df_by_continent = filtered_df[filtered_df['continent'] == i]
#         traces.append(go.Scatter(
#             x=df_by_continent['gdpPercap'],
#             y=df_by_continent['lifeExp'],
#             text=df_by_continent['country'],
#             mode='markers',
#             opacity=0.7,
#             marker={
#                 'size': 15,
#                 'line': {'width': 0.5, 'color': 'white'}
#             },
#             name=i
#         ))

#     return {
#         'data': traces,
#         'layout': go.Layout(
#             xaxis={'type': 'log', 'title': 'GDP Per Capita'},
#             yaxis={'title': 'Life Expectancy', 'range': [20, 90]},
#             margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
#             legend={'x': 0, 'y': 1},
#             hovermode='closest'
#         )
#     }

if __name__ == '__main__':
    app.run_server(debug=True)

