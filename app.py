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
FROM df_merge_classify_final;
"""
df_classify = pd.read_sql_query(sql_query,con)

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

def high_risk_func():
	return 'GENERIC POSES HIGH RISK'
def low_risk_func():
	return 'GENERIC POSES LOW RISK'


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

app.layout = html.Div(style={'backgroundColor': '#f4f5f8'}, children=[
    html.Div([
	    # Page Header
	    html.H1(
	    	children = 'Generic''s For Geriatrics',
	    	style = {
	    		'textAlign' : 'center',
	    		'color' : colors_banner['text']
	    	}
		),
	    
	    # Page Sub Header
	    html.H3(
	    	children = 'Assessing Generic Drug Risk',
	    	style = {
	    		'textAlign' : 'center',
	    		'color' : colors_banner['text'],
	    		'fontSize' : 25
	    	}
		)
	]),

	html.Br(),
	html.Br(),

    # Select generic Dropdown
    html.Div([
        html.Div('Select Generic Drug', className='three columns', style={'position': 'absolute', 'left': 125, 'top': 185}),
        html.Div(dcc.Dropdown(id='generic-selector',
                              options=onLoad_generic_options()),
                 className='one columns',
	    		 style={'position': 'absolute', 'left': 250, 'top': 180, "width" : "50%"})
    ], className="row"),

	html.Br(),
	html.Br(),

    # Page Sub Header
	# html.Div([
	# 	html.H3(id='output-risk'), style={'textAlign' : 'center', 'color': 'red'}),
	# ], className="row"),

	html.Div(id='output-risk', className="row"),

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

	html.Br(),
	html.Br(),

	    # Page Sub Header
    html.H3(
    	children = 'Basic Drug Information',
    	style = {
    		'textAlign' : 'center',
    		'color' : colors_banner['text'],
    		'fontSize' : 25
    	}
	),
    html.Div(id='generic-information', style={'textAlign' : 'center'})

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
# def model_risk_value(value):
#     message = list()
#     if value is not None:
#     	df_classify_generic = df_classify[df_classify['generic_name'] == value]
#     	p = df_classify_generic['classify_risk']
#     	risk = 'GENERIC POSES HIGH RISK' if p.all() == 0 else 'GENERIC POSES LOW RISK'
#     	print(p)
#     	return '{}'.format(risk)
def model_risk_value(value):
    message = list()
    if value is not None:
    	df_classify_generic = df_classify[df_classify['generic_name'] == value]
    	p = df_classify_generic['classify_risk']
    	risk = 'GENERIC POSES HIGH RISK' if p.all() == 0 else 'GENERIC POSES LOW RISK'
    	color_output = 'red' if p.all() == 0 else '#75a0c7'
    	return html.Div('{}'.format(risk), style={'textAlign' : 'center', 'color' : color_output, 
    		'fontSize' : 20})



# Display the price difference
@app.callback(
    dash.dependencies.Output('price_compare', 'figure'),
    [dash.dependencies.Input('generic-selector', 'value')])
def update_output(value):
    filtered_df = df[df['generic_name'] == value]
    price_min_max = filtered_df['min_price_per_dose']
    if value is not None:
	    return {
	        'data': [go.Bar(
	            x=filtered_df['brand_name'],
	            y=filtered_df['max_price_per_dose'],
	            opacity=0.7,
	        )],
	        'layout': go.Layout(
	            xaxis={'title': 'Drug Name'},
	            yaxis={'title': 'Price ($)'},
	            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
	            #legend={'x': 0, 'y': 1},
	            hovermode='closest'
	        )
	        # 'layout': {
	        #     'plot_bgcolor': '#f4f5f8',
	        #     'paper_bgcolor': '#f4f5f8',
	        #     'opacity' : 0.7,
	        #     'height': 450,
	        #     'margin': {'l': 20, 'b': 30, 'r': 10, 't': 10},
	        #     'yaxis': {'title': 'Price ($)', 'type': 'linear'},
	        #     'xaxis': {'title' : 'Drug Name', 'showgrid': False}
	        # }
	    }

# Display the number of patients on drug difference
@app.callback(
    dash.dependencies.Output('patients_on_drug', 'figure'),
    [dash.dependencies.Input('generic-selector', 'value')])
def update_output(value):
    filtered_df = df[df['generic_name'] == value]
    if value is not None:
	    return {
	        'data': [go.Bar(
	            x=filtered_df['brand_name'],
	            y=filtered_df['total_beneficiaries'],
	            opacity=0.7,
	        )],
	        'layout': go.Layout(
	            xaxis={'title': 'Drug Name'},
	            yaxis={'type': 'log', 'title': '# Patients'},
	            margin={'l': 40, 'b': 40, 't': 10, 'r': 10},
	            #legend={'x': 0, 'y': 1},
	            hovermode='closest'
	        )
	    }

@app.callback(
    dash.dependencies.Output('generic-information', 'children'),
    [dash.dependencies.Input('generic-selector', 'value')])
def generic_drug_table(value):
	df_generic = df_classify[df_classify['generic_name'] == value]
	df_sub = df_generic[['generic_name', 'total_manuf', 'increase_manuf', 'num_act_ingredients', 'nti_index']]
	df_sub.rename(columns={'generic_name':'Generic Name', 'total_manuf' : 'Total # Manufacturers',
		'increase_manuf' : 'Increase in Manufacturers', 'num_act_ingredients' : 'Number of Active Ingredients',
		'nti_index' : 'Narrow Therapeutic Index'}, inplace=True)
	if value is not None:
		return html.Div(children=[
			generate_table(df_sub)])
		# return html.Div(children=[
  #               html.Table(
  #                   # Header
  #                   [html.Tr([html.Th(col) for col in disp_columns])] +
  #                   # Body
  #                   [html.Tr([
  #                       html.Td(wl.df[wl.df.index==idx][col]) for col in df_columns
  #                   ]) for idx in indexes]
  #               ),

  #           ]
  #       )
# 'Total # Manufacturers', 'Increase in Manufacturers (per year)',
#	 'Number of Active Ingredients', 'Narrow Therapeutic Index'
		# 	generate_table(df_sub)
		
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

