# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Show table from database
import time
#from sqlalchemy import create_engine
#from sqlalchemy_utils import database_exists, create_database
#import psycopg2
import pandas as pd
import plotly.graph_objs as go


########## QUERIES ##########
# Define a database name (we're using a dataset on births, so we'll call it birth_db)
# Set your postgres username
# dbname = 'fda_adverse_events'
# username = 'pami' # change this to your username

### QUERYING THE DATABASE ###
# Connect to make queries using psycopg2
# con = None
# con = psycopg2.connect(database = dbname, user = username)
# query:
# sql_query = """
# SELECT *
# FROM df_merge_classify_final;
# """
df_classify = pd.read_pickle("./data/df_merge_classify_final.pkl")

# sql_query = """
# SELECT *
# FROM df_spending_adverse_total;
# """
df = pd.read_pickle("./data/df_merge_ad_spending.pkl")

# sql_query = """
# SELECT *
# FROM df_patient_react_2013_table;
# """
df_patient_react = pd.read_pickle("./data/df_patient_react.pkl")

# def fetch_data(q):
# 	result = pd.read_sql(
# 		sql=q,
# 		con=con
# 	)
# 	return result

def get_generics():
	'''Returns the list of generics that are stored in the database'''

	# generic_query = (
	# 	f'''
	# 	SELECT DISTINCT(generic_name)
	# 	FROM df_spending_adverse_total;
	# 	'''
	# )
	# generics = fetch_data(generic_query)
	generics = pd.read_pickle("./data/df_merge_ad_spending.pkl")
	generics = generics['generic_name']
	generics = list(generics.sort_values(ascending=True))
	return generics

def onLoad_generic_options():
	'''Actions to perform upon initial page load'''

	generic_options = (
		[{'label': generic, 'value': generic}
		 for generic in get_generics()]
	)
	return generic_options


########## FUNCTIONS FOR TABLES AND GRAPHS ##########
def generate_table(dataframe, max_rows=5):
	return html.Table(
		# Header
		[html.Tr([html.Th(col) for col in dataframe.columns])] +

		# Body
		[html.Tr([
			html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
		], style={'textAlign': 'center'}) for i in range(min(len(dataframe), max_rows))]
	)

########## DASH APP ##########
#app = dash.Dash(name)
app = dash.Dash(__name__)
app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

## CUSTOM COLOR CONFIG ##
colors_light = {
	'background': '#f4f5f8',
	'text': '#1e2832'
}

app.layout = html.Div(style={'backgroundColor': colors_light['background']}, children=[
	html.Div([
		# Page Header
		html.Div(
			className="app-header",
			children=[
				html.Div('Generic''s For Geriatrics', className="app-header--title")
			]
		),
	]),

	html.Div([
		html.Img(src='/assets/pill_background.jpg',  style={'width': '100%', 'height': '100%'}),
		html.Div(
			children = 'Assessing Generic Drug Risk',
			style = {
				'position': 'absolute',
				'left': 15, 'top': 75,
				'textAlign' : 'left',
				'color' : colors_light['text'],
				'fontSize' : 20, 
				'backgroundColor': 'transparent'
			}
		)
	], className="row"),

	# Select generic Dropdown
	html.Div([
		html.Div('Select Generic Drug', className='three columns', style={'position': 'absolute', 'left': 125, 'top': 235}),
		html.Div(dcc.Dropdown(id='generic-selector',
							  options=onLoad_generic_options()),
				 className='one columns',
				 style={'position': 'absolute', 'left': 250, 'top': 230, "width" : "50%"})
	], className="row"),

	# Risk output
	html.Div(id='output-risk', className="row"),

	# Blank spaces
	html.Br(),

	# Table wit Adverse events
	html.Div(id='generic-adr', style={'position' : 'relative', 'left': '40%', 'width': '50%'}),

	html.Br(),
	html.Br(),

	# Display the plots for the selected drug
	html.Div([
		html.Div([
			dcc.Graph(id='patients_on_drug')
		], className="six columns"),

		html.Div([
			dcc.Graph(id='price_compare')
		], className="six columns"),
	], className="row"),

	html.Br(),
	html.Br(),

	# Table with Drug information
	html.H6(
		children = 'Basic Drug Information',
		style = {
			'textAlign' : 'center',
			'color' : colors_light['text'],
			'fontSize' : 25
		}
	),
	html.Div(id='generic-information', style={'textAlign' : 'center'})

])

# Display generic risk
@app.callback(
	dash.dependencies.Output('output-risk', 'children'),
	[dash.dependencies.Input('generic-selector', 'value')])
def model_risk_value(value):
	message = list()
	if value is not None:
		df_classify_generic = df_classify[df_classify['generic_name'] == value]
		p = df_classify_generic['classify_risk']
		risk = 'GENERIC POSES HIGH RISK' if p.all() == 0 else 'GENERIC POSES LOW RISK'
		# color_output = 'red' if p.all() == 0 else '#75a0c7'
		if p.all() == 0:
			color_output = 'red'
			text_color = colors_light['background']
		else:
			color_output = '#1e2832'
			text_color = colors_light['background']
		return html.H3('{}'.format(risk), style={'textAlign' : 'center', 'backgroundColor' : color_output, 
			'fontSize' : 30, 'color': text_color, 'width': '50%', 
			'position': 'relative', 'left': '25%', "width" : "50%"})

@app.callback(
	dash.dependencies.Output('generic-adr', 'children'),
	[dash.dependencies.Input('generic-selector', 'value')])
def generic_adr_table(value):
	df_generic = df_patient_react[df_patient_react['drug_generic_name'] == value]
	print(df_generic.columns.values)
	if df_generic.empty:
		txt_disp = 'No Adverse Drug Reactions Reported Last 2 Years'
		df_sub = pd.DataFrame()
	else:
		txt_disp = 'Adverse Drug Reactions'
		df_generic = df_generic.sort_values(by='reaction', ascending=False)
		df_sub = df_generic[['patient_react_type']]
		df_sub.rename(columns={'patient_react_type' : 'Top Five Most Common'}, inplace=True)
	if value is not None:
		return html.Div(children=[
				html.H5(
					children = txt_disp,
					style = {
						'color' : colors_light['text'],
						'position' : 'relative', 'width': '50%', 'textAlign' : 'center'
					}),
				generate_table(df_sub)]) #, style={'position' : 'relative', 'right': '70%'})

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
				marker={'color':'green'}
			)],
			'layout': go.Layout(
				title='Price Comparison',
				xaxis={'title': 'Drug Name'},
				yaxis={'title': 'Price ($)'},
				margin={'l': 40, 'b': 70, 't': 40, 'r': 10},
				paper_bgcolor=colors_light['background'],
				plot_bgcolor=colors_light['background'],
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
				marker={'color':'blue'}
			)],
			'layout': go.Layout(
				title='# Patients on Drug',
				xaxis={'title': 'Drug Name'},
				yaxis={'type': 'log', 'title': '# Patients'},
				margin={'l': 70, 'b': 70, 't': 40, 'r': 10},
				paper_bgcolor=colors_light['background'],
				plot_bgcolor=colors_light['background'],
				hovermode='closest'
				#legend={'x': 0, 'y': 1},
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


if __name__ == '__main__':
	app.run_server(debug=True)
# if __name__ == '__main__':
# 	app.run_server(host='0.0.0.0', debug=True)

