import dash
from dash import dcc
from dash import html
import plotly.graph_objs as go
from dash.dependencies import Input, Output
import external_stock_data
import prediction
from datetime import date
import pandas as pd


def calculate_roc(data):
    roc = (data - data.shift(1)) / data.shift(1) * 100
    return roc


# init app
app = dash.Dash()
server = app.server

# implement ui
app.layout = html.Div([
   
   # title
    html.H1("Stock Price Analysis Dashboard", style={"textAlign": "center"}),

    # tool bar
    html.Div(
        style={"display": "flex", "gap": "20px", "align-items": "center"},
        children=[
            dcc.Dropdown(
                id='coin-dropdown',
                options=['BTC-USD', 'ETH-USD', 'ADA-USD'], 
                value='BTC-USD', 
                clearable=False,
                style={"width": "200px"}),
            dcc.Dropdown(
                id='price-type-dropdown',
                options=[
                    {'label': 'Open Price', 'value': 'Open'},
                    {'label': 'Close Price', 'value': 'Close'},
                    {'label': 'Low Price', 'value': 'Low'},
                    {'label': 'High Price', 'value': 'High'},
                    {'label': 'ROC', 'value': 'ROC'},
                ], 
                value='Open', 
                clearable=False,
                style={"width": "200px"}),
            dcc.Dropdown(
                id='algorithm-dropdown',
                options=[
                    {'label': 'LSTM', 'value': 'lstm'},
                    {'label': 'RNN', 'value': 'rnn'},
                    {'label': 'XGBoost', 'value': 'xgboost'},
                ], 
                value='lstm', 
                clearable=False,
                style={"width": "200px"}),

            html.H5("Start Date:",style={"margin-left": "20px"}),
            dcc.DatePickerSingle(
                id='start-date',
                min_date_allowed=date(2018, 1, 1),
                max_date_allowed=date.today(),
                initial_visible_month=date(2020, 1, 1),
                date=date(2020, 1, 1),
                clearable=False,
                display_format="DD/MM/YYYY"
            ),

            html.H5("End Date:"),
            dcc.DatePickerSingle(
                id='end-date',
                min_date_allowed=date(2018, 1, 1),
                max_date_allowed=date.today(),
                initial_visible_month=date.today(),
                date=date.today(),
                clearable=False,
                display_format="DD/MM/YYYY",
            ),
    ]),

    # data presention by graph
    html.Div(
        children = [
            html.H2("Actual And Predicted Prices",style={"textAlign": "center"}),
            dcc.Loading(
                dcc.Graph(id="price-graph"),
            ),

            html.H2("Transactions Volume",style={"textAlign": "center"}),
            dcc.Loading(
                dcc.Graph(id="volume-graph")				
            ),
        ],
        style={"border": "solid 1px gray", "marginTop": "10px"}  
    ),
])

# update price graph follow by input user
@app.callback(Output('price-graph', 'figure'),
              [
                  Input('coin-dropdown', 'value'),
                  Input('price-type-dropdown', 'value'),
                  Input('algorithm-dropdown', 'value'),
                  Input('start-date', 'date'),
                  Input('end-date', 'date')
              ])
def update_price_graph(coin, price_type, algorithm, start_date, end_date):
    # pick range data to predict
    # start_date = '2023-01-20'
    # end_date = date.today()

    #choose model to predict
    if algorithm == 'rnn':
        print('runing RNN algorithm')
        predPrice = prediction.predictByRNN(coin, price_type, start_date, end_date)
    elif algorithm == 'xgboost':
        print('runing XGBoost algorithm')
        predPrice = prediction.predictByXGBoost(coin, price_type, start_date, end_date)
    else:
        print('runing default algorithm: LSTM')
        predPrice = prediction.predictByLSTM(coin, price_type, start_date, end_date)

    # present data to graph
    dataPrice = external_stock_data.getStockData(coin, start_date, end_date)
    dataPrice["ROC"] = calculate_roc(dataPrice["Close"])
    figure = {
        'data': [
            go.Scatter(
                x=dataPrice.index,
                y=dataPrice[price_type],
                mode='lines',
                opacity=0.7, 
                name=f'Actual {price_type} Price',textposition='bottom center'),
            go.Scatter(
                x=predPrice.index,
                y=predPrice["Predictions"],
                mode='lines',
                opacity=0.6,
                name=f'Predicted {price_type} Price',textposition='bottom center')
        ],
        'layout': go.Layout(colorway=["#5E0DAC", '#FF4F00', '#375CB1', 
                                        '#FF7400', '#FFF400', '#FF0056'],
        height=600,
        yaxis={"title": "ROC (%)" if price_type == "ROC" else "Price (USD)"})
    }
    return figure

# update volume graph follow by input user
@app.callback(Output('volume-graph', 'figure'),
              [
                Input('coin-dropdown', 'value'),
                Input('start-date', 'date'),
                Input('end-date', 'date')
              ])
def update_volume_graph(coin, start_date, end_date):
    # pick range data to predict
    # start_date = '2023-01-20'
    # end_date = date.today()
    dataVolume = external_stock_data.getStockData(coin, start_date, end_date)
    figure = {
        'data': [
            go.Bar(
                x=dataVolume.index,
                y=dataVolume["Volume"],
                opacity=0.7,
                marker=dict(color='green'),
                name=f'Volume'),
        ], 
        'layout': go.Layout(
            height=600,
            yaxis={"title":"Volume"})
    }
    return figure

# start app
if __name__=='__main__':
	app.run_server(debug=True)