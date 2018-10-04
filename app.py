# -*- coding: utf-8 -*-
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output

# Show table from database
import time
import pandas as pd
import numpy as np
import plotly.graph_objs as go


########## QUERIES ##########
df = pd.read_pickle("./data/df_merge_ad_spending.pkl")
df_classify = pd.read_pickle("./data/df_merge_classify_final.pkl")
df_patient_react = pd.read_pickle("./data/df_patient_react.pkl")

def get_generics():
	'''Returns the list of generics that are stored in the database'''
	generics = df_classify['generic_name']
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
def generate_table(dataframe, max_rows=5, style_parameters=None):
	return html.Table(
		# Header
		[html.Tr([html.Th(col) for col in dataframe.columns])] +

		# Body
		[html.Tr([
			html.Td(dataframe.iloc[i][col]) for col in dataframe.columns
		], style=style_parameters) for i in range(min(len(dataframe), max_rows))]
	)

########## DASH APP ##########
app = dash.Dash(__name__)
app.title = "Generics for Geriatrics"
#app.css.append_css({"external_url": "https://codepen.io/chriddyp/pen/bWLwgP.css"})

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
		html.Img(src='/assets/pill_background_6.jpg',  style={'width': '100%', 'height': '100%'}),
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
		html.Div('Select Generic Drug', className='three columns', style={'position': 'absolute', 'left': '30%', 'top': '40%'}),
		html.Div(dcc.Dropdown(id='generic-selector',
							  options=onLoad_generic_options()),
				 className='one columns',
				 style={'position': 'absolute', 'left': '25%', 'top': '43%', "width" : "50%"})
	], className="row"),

	# Risk output
	html.Div(id='output-risk', className="row"),
	html.Div(id='output-risk-text', className="row"),

	# Blank spaces
	html.Br(),

	# Table wit Adverse events
	html.Div(id='generic-adr'), #style={'position' : 'relative', 'left': '45%', 'width': '50%'}

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
		risk_class = df_classify_generic['classify_risk']
		type_count = df_classify_generic['risk_count']
		type_class = df_classify_generic['risk_class']
		if type_count.all() == 1 & type_class.all() == 1:
			risk = 'ONLY BRAND AVAILABLE'
			color_output = '#1e2832'
			text_color = colors_light['background']
		elif type_count.all() == 1 & type_class.all() == -1:
			risk = 'ONLY GENERIC AVAILABLE'
			color_output = '#1e2832'
			text_color = colors_light['background']
		else:
			if risk_class.all() == 1:
				risk = 'GENERIC POSES LOW RISK'
				color_output = '#1e2832'
				text_color = colors_light['background']
			else:
				risk = 'GENERIC POSES HIGH RISK'
				color_output = 'red'
				text_color = colors_light['background']

		return html.H3('{}'.format(risk), style={'textAlign' : 'center', 'backgroundColor' : color_output, 
			'color': text_color, 'width' : '50%', 'position' : 'relative', 'left': '25%'})

# Display text about generic risk
@app.callback(
	dash.dependencies.Output('output-risk-text', 'children'),
	[dash.dependencies.Input('generic-selector', 'value')])
def model_risk_value(value):
	message = list()
	if value is not None:
		df_classify_generic = df_classify[df_classify['generic_name'] == value]
		p = df_classify_generic['classify_risk']
		# color_output = 'red' if p.all() == 0 else '#75a0c7'
		if p.all() == 0:
			color_output = 'red'
			text_color = colors_light['text']
			text_output_1 = 'This generic drug poses additional risks of adverse drug reactions'
			text_output_2 = 'due to previous adverse events,\n number of manufacturers, and'
			text_output_3 = 'spending information \n from the FDA and CMS.'
		else:
			text_output_1 = 'This generic drug does not pose additional risks of adverse drug'
			text_output_2 = 'reactions and shows a minimal increase in adverse events.'
			text_output_3 = ''
			text_color = colors_light['text']
		return (html.H6('{}'.format(text_output_1), style={'textAlign' : 'center'}),
				html.H6('{}'.format(text_output_2), style={'textAlign' : 'center'}),
				html.H6('{}'.format(text_output_3), style={'textAlign' : 'center'}))

@app.callback(
	dash.dependencies.Output('generic-adr', 'children'),
	[dash.dependencies.Input('generic-selector', 'value')])
def generic_adr_table(value):
	df_generic = df_patient_react[df_patient_react['drug_generic_name'] == value]
	style_parameters = style={'width' : '50%', 'position' : 'relative', 
								'float': 'center', 'position': 'relative', 'left' : '40%'}
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
						'position': 'relative', 'left' : '0%'
					}),
				html.Div(generate_table(df_sub), style=style_parameters)])

# Display the price difference
@app.callback(
	dash.dependencies.Output('price_compare', 'figure'),
	[dash.dependencies.Input('generic-selector', 'value')])
def update_output(value):
	filtered_df = df[df['generic_name'] == value]
	price_min_max = filtered_df['min_price_per_dose']
	
	filter_colors = filtered_df['risk_class'].copy()
	filter_colors[filter_colors == -1] = 'rgb(0,0,255)'
	filter_colors[filter_colors == 1] = 'rgb(255,0,0)'

	filter_colors_leg = filtered_df['risk_class'].copy()
	filter_colors_leg[filter_colors_leg == -1] = 'Generic'
	filter_colors_leg[filter_colors_leg == 1] = 'Brand'

	if value is not None:
		return {
			'data': [go.Bar(
				x=filtered_df['brand_name'],
				y=filtered_df['max_price_per_dose'],
				opacity=0.9,
				marker=dict(color=filter_colors.tolist()),
				text=filter_colors_leg.tolist(),
				textposition = 'auto',
				textfont=dict(
					family='sans serif',
					size=14,
					color='#ffffff'
				)
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
		}

# Display the number of patients on drug difference
@app.callback(
	dash.dependencies.Output('patients_on_drug', 'figure'),
	[dash.dependencies.Input('generic-selector', 'value')])
def update_output(value):
	filtered_df = df[df['generic_name'] == value]
	
	filter_colors = filtered_df['risk_class'].copy()
	filter_colors[filter_colors == -1] = 'rgb(0,0,255)'
	filter_colors[filter_colors == 1] = 'rgb(255,0,0)'

	filter_colors_leg = filtered_df['risk_class'].copy()
	filter_colors_leg[filter_colors_leg == -1] = 'Generic'
	filter_colors_leg[filter_colors_leg == 1] = 'Brand'

	if value is not None:
		return {
			'data': [go.Bar(
				x=filtered_df['brand_name'],
				y=filtered_df['total_beneficiaries'],
				opacity=0.9,
				marker=dict(color=filter_colors.tolist()),
				text=filter_colors_leg.tolist(),
				textposition = 'auto',
				textfont=dict(
					family='sans serif',
					size=14,
					color='#ffffff'
				)
				#name= dict(name=filter_colors_leg.tolist())
				#marker={'color':'blue'}
			)],
			'layout': go.Layout(
				title='# Patients on Drug',
				xaxis={'title': 'Drug Name'},
				yaxis={'type': 'log', 'title': '# Patients'},
				margin={'l': 70, 'b': 70, 't': 40, 'r': 10},
				paper_bgcolor=colors_light['background'],
				plot_bgcolor=colors_light['background'],
				hovermode='closest',
				# legend={'x': 0, 'y': 1},
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

