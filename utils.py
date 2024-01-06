import requests
from bs4 import BeautifulSoup
import pandas as pd
import numpy as np
import joblib
from scipy.stats import boxcox
from datetime import datetime
import plotly.express as px
import plotly.subplots as sp
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
from statsmodels.tsa.seasonal import seasonal_decompose
from statsmodels.tsa.stattools import adfuller, acf, pacf

class DataExtractor:
    def __init__(self, url = "http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view"):
        self.url = url

    def extract_data(self):
        save = saver()
        response = requests.get(self.url)
        html_content = response.content
        soup = BeautifulSoup(html_content, 'html.parser')
        element_center = soup.find('body').find('form').find_all('center') #type: ignore
        second_center = element_center[1]
        lines = second_center.find_all('tr')

        dates = []
        prices = []

        for line in lines[1:]:
            cell = line.find_all('td')

            if len(cell) >= 2:
                data = cell[0].get_text(strip=True)
                price = cell[1].get_text(strip=True)

                dates.append(data)
                prices.append(price)

        df = pd.DataFrame({'Data': dates, 'Preco': prices})
        df = df.iloc[3:]
        df = df.iloc[:-2]

        save.save_parquet(df, 'dados.parquet', False)

        df = df.rename(columns={'Data': 'Data', 'Preco': 'y'})
        df = df.set_index('Data')

        df['y'] = df['y'].str.replace(',', '.').astype('float64')
        df.index = pd.to_datetime(df.index, dayfirst=True)

        save.save_parquet(df, 'dados_formatados.parquet', True)

class saver:
    def save_parquet(self, df, filename, index):
        df.to_parquet(filename, index=index)

class Loader:
    def load_data(self, filename='dados.parquet'):
        return pd.read_parquet(filename)
    
    def load_data_trans(self, filename='dados_formatados.parquet'):
        return pd.read_parquet(filename)
    
    def load_data_diff(self, filename='dados_diferenciados.parquet'):
        return pd.read_parquet(filename)
    
    def load_data_error(self, filename='error.parquet'):
        return pd.read_parquet(filename)
    
    def load_data_train(self, filename='train.parquet'):
        return pd.read_parquet(filename)
    
    def load_data_test(self, filename='test.parquet'):
        return pd.read_parquet(filename)
    
    def load_data_valid(self, filename='valid.parquet'):
        return pd.read_parquet(filename)

class TimeSeriesPlotter:
    def plot(self, df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['y'], mode='lines', line=dict(color='lightblue')))
        
        fig.update_layout(
            title='Variações do preço por barril do petróleo bruto tipo Brent',
            xaxis=dict(title='Data'),
            yaxis=dict(title='Valor'),
            showlegend=False,
            height=600
        )

        return fig

    def plot_with_events(self, df):
        fig = go.Figure()
        fig.add_trace(go.Scatter(x=df.index, y=df['y'], mode='lines', line=dict(color='lightblue')))

        events = [
            {'date': '1990-07-01', 'description': 'Guerra do Golfo'},
            {'date': '2001-09-11', 'description': 'Ataques de 11 de setembro'},
            {'date': '2008-07-01', 'description': 'Crise financeira global'},
            {'date': '2014-07-01', 'description': 'Excesso de oferta global de petróleo'},
            {'date': '2020-04-01', 'description': 'COVID-19'},
            {'date': '2022-06-01', 'description': 'Rússia-Ucrânia'}
        ]

        for event in events:
            event_date = datetime.strptime(event['date'], '%Y-%m-%d')
            fig.add_shape(
                go.layout.Shape(
                    type='line',
                    x0=event_date,
                    x1=event_date,
                    y0=0,
                    y1=1,
                    yref='paper',
                    line=dict(color='lightcoral', width=2),
                    name=event['description']
                )
            )
        
            fig.add_annotation(
                go.layout.Annotation(
                    x=event_date,
                    y=1.1,
                    xref='x',
                    yref='paper',
                    text=event['description'],
                    showarrow=False,
                    font=dict(color='lightcoral', size=10),
                )
            )

        fig.update_layout(
            title='Variações do preço por barril do petróleo bruto tipo Brent',
            xaxis=dict(title='Data'),
            yaxis=dict(title='Valor'),
            showlegend=False,
            height=600
        )

        return fig

    def decompose(self, df):
        results = seasonal_decompose(df['y'], model='additive', period=365, two_sided=True, extrapolate_trend=5)
        fig_observed = px.line(results.observed, x=results.observed.index, y='y', title='Série Observada')
        fig_observed.update_xaxes(title_text='Data')
        fig_trend = px.line(results.trend, x=results.trend.index, y='trend', title='Componente de Tendência')
        fig_trend.update_xaxes(title_text='Data')
        fig_seasonal = px.line(results.seasonal, x=results.seasonal.index, y='seasonal', title='Componente de Sazonalidade')
        fig_seasonal.update_xaxes(title_text='Data')
        fig_resid = px.line(results.resid, x=results.resid.index, y='resid', title='Resíduo')
        fig_resid.update_xaxes(title_text='Data')
        fig = sp.make_subplots(
            rows=4,
            cols=1,
            shared_xaxes=True,
            vertical_spacing=0.1, 
            subplot_titles=[
                'Série Observada',
                'Componente de Tendência',
                'Componente de Sazonalidade',
                'Resíduo']
        )
        fig.add_trace(go.Scatter(x=fig_observed.data[0]['x'], y=fig_observed.data[0]['y'], showlegend=False), row=1, col=1) # type: ignore
        fig.add_trace(go.Scatter(x=fig_trend.data[0]['x'], y=fig_trend.data[0]['y'], showlegend=False), row=2, col=1) # type: ignore
        fig.add_trace(go.Scatter(x=fig_seasonal.data[0]['x'], y=fig_seasonal.data[0]['y'], showlegend=False), row=3, col=1) # type: ignore
        fig.add_trace(go.Scatter(x=fig_resid.data[0]['x'], y=fig_resid.data[0]['y'], showlegend=False), row=4, col=1) # type: ignore
        fig.update_xaxes(title_text='Data', row=4, col=1)
        fig.update_layout(title='Decomposição da Série Temporal - Preço do Barril', font=dict(size=12))
        fig.update_layout(
            height=600
        )
        return fig
    
    def autocorrelation_function(self, df, lag):
        return acf(df.dropna(), nlags=lag)

    def partial_autocorrelation_function(self, df, lag):
        return pacf(df.dropna(), nlags=lag)

    def acf_pacf(self, df):
        fig = sp.make_subplots(rows=1, cols=2, subplot_titles=['ACF - Preço do Barril', 'PACF - Preço do Barril'])
        trace_acf = go.Scatter(x=np.arange(len(self.autocorrelation_function(df, 14))),
                            y=self.autocorrelation_function(df, 14),
                            mode='lines',
                            name='ACF')
        trace_pacf = go.Scatter(x=np.arange(len(self.partial_autocorrelation_function(df, 14))),
                                y=self.partial_autocorrelation_function(df, 14),
                                mode='lines',
                                name='PACF')
        fig.add_trace(trace_acf, row=1, col=1)
        fig.add_trace(trace_pacf, row=1, col=2)
        hline = go.Scatter(x=[0, len(df) - 1],
                        y=[-1.96 / (np.sqrt((len(df) - 1))), -1.96 / (np.sqrt((len(df) - 1)))],
                        mode='lines',
                        line=dict(color='gray', dash='dash'),
                        showlegend=False)
        fig.add_trace(hline, row=1, col=1)
        fig.add_trace(hline, row=1, col=2)
        hline = go.Scatter(x=[0, len(df) - 1],
                        y=[1.96 / (np.sqrt((len(df) - 1))), 1.96 / (np.sqrt((len(df) - 1)))],
                        mode='lines',
                        line=dict(color='gray', dash='dash'),
                        showlegend=False)
        fig.add_trace(hline, row=1, col=1)
        fig.add_trace(hline, row=1, col=2)
        q_value = 0.85
        p_value = 0.85
        point_acf = go.Scatter(x=[q_value], y=[0.017], mode='markers', marker=dict(color='red'), name='Q')
        point_pacf = go.Scatter(x=[p_value], y=[0.017], mode='markers', marker=dict(color='red'), name='P')
        fig.add_trace(point_acf, row=1, col=1)
        fig.add_trace(point_pacf, row=1, col=2)
        fig.update_xaxes(title_text='Lag', range=[0, 15], row=1, col=1)
        fig.update_xaxes(title_text='Lag', range=[0, 15], row=1, col=2)
        fig.update_yaxes(title_text='Correlação', range=[-0.30, 0.3], row=1, col=1)
        fig.update_yaxes(title_text='Correlação', range=[-0.30, 0.3], row=1, col=2)
        fig.update_layout(title='ACF e PACF do Preço do Barril', showlegend=True)
        
        return fig
    
    def plot_acf_pacf(self, df):
        df = df.rename(columns={'Data': 'Data', 'Preco': 'y'})
        df = df.set_index('Data')
        df['y'] = df['y'].str.replace(',', '.').astype('float64')
        df.index = pd.to_datetime(df.index, dayfirst=True)
        fig, ax = plt.subplots(1, 2, figsize=(16, 6))
        
        plot_acf(df, ax=ax[0])
        ax[0].set_title('ACF - Preço do Barril')
        ax[0].plot(1, 1.96/np.sqrt(len(df['y']) - 1), 'ro')

        plot_pacf(df['y'], ax=ax[1])
        ax[1].set_title('PACF - Preço do Barril')
        ax[1].plot(1, 1.96/np.sqrt(len(df['y']) - 1), 'ro')

        return fig
    
    def plot_error(self, df):
        df_sorted = df.sort_values(by='wmape_traino', ascending=True)
        num_models = len(df_sorted)
        custom_color_palette = ['#1f77b4', '#ff7f0e', '#2ca02c', '#d62728', '#9467bd', '#8c564b',
                                '#e377c2', '#7f7f7f', '#bcbd22', '#17becf', '#aec7e8', '#ffbb78',
                                '#98df8a', '#ff9896', '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7',
                                '#dbdb8d', '#9edae5', '#aec7e8', '#ffbb78', '#98df8a', '#ff9896',
                                '#c5b0d5', '#c49c94', '#f7b6d2', '#c7c7c7']

        color_scale = custom_color_palette[:num_models][::-1]

        fig = px.bar(df_sorted, x='model', y=['wmape_traino', 'wmape_validacao'],
                    labels={'value': 'wMAPE', 'variable': 'Conjunto de Dados', 'model': 'Modelo'},
                    title="wMAPE (Weighted Mean Absolute Percentage Error)",
                    color_discrete_sequence=color_scale)

        fig.update_layout(
            barmode='group',
            yaxis_tickformat=".2%",
            yaxis_title='wMAPE',
            xaxis_title="Modelo",
            xaxis_categoryorder='total ascending',
            margin=dict(l=0, r=0, t=50, b=0),
            showlegend=True
        )
        fig.update_yaxes(showgrid=True, gridwidth=1, gridcolor='lightgray')
        fig.update_xaxes(showgrid=False)
        fig.update_layout(
            height=400
        )
        return fig

class Differentiation:
    def data_diff(self, df, df_mean, df_std, title):
        trace_diff = go.Scatter(x=df.index, y=df, mode='lines', name='Variação')
        trace_mean = go.Scatter(x=df_mean.index, y=df_mean, mode='lines', name='Média', line=dict(color='red'))
        trace_std = go.Scatter(x=df_std.index, y=df_std, mode='lines', name='Desvio Padrão', line=dict(color='green'))
        fig = go.Figure(data=[trace_diff, trace_mean, trace_std])
        fig.update_xaxes(title_text='Data')
        fig.update_yaxes(title_text='Preço do Barril')
        fig.update_layout(title=f'Variação, Média e Desvio Padrão do Preço do Barril: {title}', showlegend=True)

        return fig

class FillNulls:
    def rolling_mean(self, df, index='Data', column_ds='ds', column_y='y', window_size=6):
        save = saver()
        date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
        df = df.reindex(date_range)
        rolling_mean = df[column_y].rolling(window=window_size, min_periods=1).mean()
        is_null_sequence = df[column_y].isnull()
        df[column_y] = np.where(is_null_sequence, rolling_mean, df[column_y])
        df[column_y] = df[column_y].round(2)
        df[column_ds] = df.index
        df.index.name = index
        
        save.save_parquet(df, 'dados_formatados.parquet', True)

        return df

class Stationary:
    def make_stationary(self, df, window_size=12):
        save = saver()
        df_log = np.log(df[['y']])
        df_log_mean = df_log.rolling(window=window_size).mean()
        df_detrended = (df_log - df_log_mean).dropna()
        df_stationary = df_detrended.diff(1).dropna()

        df_stationary_diff = df_stationary['y']
        df_stationary_mean = df_stationary_diff.rolling(window=window_size).mean()
        df_stationary_std = df_stationary_diff.rolling(window=window_size).std()
        
        save.save_parquet(df_stationary, 'dados_diferenciados.parquet', True)

        return df_stationary_diff, df_stationary_mean, df_stationary_std

    def make_stationary_boxcox(self, df, window_size=12):
        data_boxcox, lambda_value = boxcox(df['y']) #type:ignore

        data_transformed = df.copy()
        data_transformed['y'] = data_boxcox

        data_diff = data_transformed['y'].diff(1).dropna()
        data_mean = data_diff.rolling(window=window_size).mean()
        data_std = data_diff.rolling(window=window_size).std()

        return data_diff, data_mean, data_std, lambda_value

class Adfuller:
    def test_adfuller(self, df):
        result = adfuller(df.values)
        return result

class Preparation:
    def data_prep(self, df_trans, df_diff):
        fill_nulls = FillNulls()
        save = saver()
        dataset = pd.DataFrame()
        dataset.index = df_diff.index
        dataset['ds'] = dataset.index
        dataset.index.name = 'Data'
        dataset['y'] = df_diff

        dataset = pd.merge(df_diff, df_trans, left_index=True, right_index=True, how='left')
        dataset.rename(columns={'y_y': 'y'}, inplace=True)
        
        while dataset.isnull().any().any():
            for column in dataset.columns:
                if dataset[column].isnull().any():
                    dataset = fill_nulls.rolling_mean(dataset, index='Data', column_ds='ds', column_y='y')


        train = dataset.loc[dataset.ds < '2023-11-01']
        test = dataset.loc[(dataset.ds >= '2023-11-01') & (dataset.ds < '2023-12-01')]
        valid = dataset.loc[dataset.ds >= '2023-12-01']

        pd.options.mode.chained_assignment = None
        train.loc[:, 'unique_id'] = 'Brent'
        test.loc[:, 'unique_id'] = 'Brent'
        valid.loc[:, 'unique_id']  = 'Brent'

        save.save_parquet(train, 'train.parquet', True)
        save.save_parquet(test, 'test.parquet', True)
        save.save_parquet(valid, 'valid.parquet', True)

class Error:
    def wmape(self, y_true, y_pred):
        wmape = np.abs(y_true - y_pred).sum() / np.abs(y_true).sum()
        return wmape
    
class Models:
    def mstl(self, train, test, valid):
        with open('MTS', 'rb') as model_file:
            MSTL_model = joblib.load(model_file)
        
        erro = Error()
        # with open('SeasonalExponentialSmoothingOptimized', 'rb') as model_file:
        #     SeasESOpt_model = joblib.load(model_file)

        MSTL_forecast = MSTL_model.predict(h=30, level=[95])
        MSTL_forecast = MSTL_forecast.reset_index().merge(test[['ds', 'y', 'unique_id']], on=['ds', 'unique_id'], how='left')
        MSTL_forecast.dropna(inplace=True)

        MSTL_forecast_valid = MSTL_model.predict(h=55, level=[95])
        MSTL_forecast_valid = MSTL_forecast_valid.reset_index().merge(valid[['ds', 'y', 'unique_id']], on=['ds', 'unique_id'], how='left')
        MSTL_forecast_valid.dropna(inplace=True)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Dados de Treinamento (1987-05-20 - 2023-11-01)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig.add_trace(go.Scatter(x=MSTL_forecast['ds'], y=MSTL_forecast['y'], mode='lines', name='Dados de Teste (2023-11-01 - 2023-12-01)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig.add_trace(go.Scatter(x=MSTL_forecast['ds'], y=MSTL_forecast['MSTL'], mode='lines', name='Previsão MSTL', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig.add_trace(go.Scatter(x=MSTL_forecast['ds'], y=MSTL_forecast['MSTL-lo-95'], mode='lines', name='Limite Inferior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig.add_trace(go.Scatter(x=MSTL_forecast['ds'], y=MSTL_forecast['MSTL-hi-95'], mode='lines', name='Limite Superior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig.add_shape(
            type='line',
            x0='2023-11-01',
            x1='2023-11-01',
            y0=10,
            y1=140,
            line=dict(color='rgba(255, 255, 0, 0.8)', dash='dash')
        )
        fig.add_annotation(
            x='2023-11-01',
            y=110,
            text='First Forecast',
            showarrow=False
        )
        fig.add_trace(
            go.Scatter(
                x=MSTL_forecast['ds'],
                y=MSTL_forecast['MSTL-lo-95'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )
        fig.update_xaxes(
            range=['2020-01-01', MSTL_forecast['ds'].max()]
        )
        
        fig1 = go.Figure()
        fig1.add_trace(go.Scatter(x=MSTL_forecast_valid['ds'], y=MSTL_forecast_valid['y'], mode='lines', name='Dados de  Validação (2023-08-16 a 2023-08-25)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig1.add_trace(go.Scatter(x=MSTL_forecast_valid['ds'], y=MSTL_forecast_valid['MSTL'], mode='lines', name='Previsão MSTL', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig1.add_trace(go.Scatter(x=MSTL_forecast_valid['ds'], y=MSTL_forecast_valid['MSTL-lo-95'], mode='lines', name='Limite Inferior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig1.add_trace(go.Scatter(x=MSTL_forecast_valid['ds'], y=MSTL_forecast_valid['MSTL-hi-95'], mode='lines', name='Limite Superior (95%)', line=dict(color='rgba(255, 0, 255, 0.8)')))
        fig1.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig1.add_trace(
            go.Scatter(
                x=MSTL_forecast_valid['ds'],
                y=MSTL_forecast_valid['MSTL-lo-95'],
                mode='lines',
                fill='tonexty',
                fillcolor='rgba(255, 0, 255, 0.1)',
                line=dict(width=0),
                showlegend=False
            )
        )
       
        
        return fig, erro.wmape(MSTL_forecast['y'].values, MSTL_forecast['MSTL'].values), fig1, erro.wmape(MSTL_forecast_valid['y'].values, MSTL_forecast_valid['MSTL'].values)
    
    def seas_es_opt(self, train, test, valid):
        with open('SeasonalExponentialSmoothingOptimized', 'rb') as model_file:
            SeasESOpt_model = joblib.load(model_file)
        
        erro = Error()

        SeasESOpt_forecast = SeasESOpt_model.predict(h=30)
        SeasESOpt_forecast = SeasESOpt_forecast.reset_index().merge(test[['ds', 'y', 'unique_id']], on=['ds', 'unique_id'], how='left')
        SeasESOpt_forecast.dropna(inplace=True)

        SeasESOpt_forecast_valid = SeasESOpt_model.predict(h=55)
        SeasESOpt_forecast_valid = SeasESOpt_forecast_valid.reset_index().merge(valid[['ds', 'y', 'unique_id']], on=['ds', 'unique_id'], how='left')
        SeasESOpt_forecast_valid.dropna(inplace=True)

        fig2 = go.Figure()
        fig2.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Dados de Treinamento (1987-05-20 - 2023-11-01)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig2.add_trace(go.Scatter(x=SeasESOpt_forecast['ds'], y=SeasESOpt_forecast['y'], mode='lines', name='Dados de Teste (2023-11-01 - 2023-12-01)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig2.add_trace(go.Scatter(x=SeasESOpt_forecast['ds'], y=SeasESOpt_forecast['SeasESOpt'], mode='lines', name='Previsão SeasESOpt', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig2.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig2.add_shape(
            type='line',
            x0='2023-11-01',
            x1='2023-11-01',
            y0=10,
            y1=140,
            line=dict(color='rgba(255, 255, 0, 0.8)', dash='dash')
        )
        fig2.add_annotation(
            x='2023-11-01',
            y=100,
            text='First Forecast',
            showarrow=False
        )
        fig2.update_xaxes(
            range=['2020-01-01', SeasESOpt_forecast['ds'].max()]
        )

        fig3 = go.Figure()
        fig3.add_trace(go.Scatter(x=SeasESOpt_forecast_valid['ds'], y=SeasESOpt_forecast_valid['y'], mode='lines', name='Dados de  Validação (2023-12-01 - 2023-12-31)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig3.add_trace(go.Scatter(x=SeasESOpt_forecast_valid['ds'], y=SeasESOpt_forecast_valid['SeasESOpt'], mode='lines', name='Previsão SeasESOpt', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig3.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )

        return fig2, erro.wmape(SeasESOpt_forecast['y'].values, SeasESOpt_forecast['SeasESOpt'].values), fig3, erro.wmape(SeasESOpt_forecast_valid['y'].values, SeasESOpt_forecast_valid['SeasESOpt'].values)
    
class BestModel:
    def mstl(self, train, h):
        with open('MTS', 'rb') as model_file:
            MSTL_model = joblib.load(model_file)
        
        MSTL_forecast = MSTL_model.predict(h=85 + h, level=[95])

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Dados de Treinamento (1987-05-20 - 2023-11-01)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig.add_trace(go.Scatter(x=MSTL_forecast['ds'], y=MSTL_forecast['MSTL'], mode='lines', name='Previsão MSTL', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig.add_shape(
            type='line',
            x0='2023-11-01',
            x1='2023-11-01',
            y0=10,
            y1=140,
            line=dict(color='rgba(255, 255, 0, 0.8)', dash='dash')
        )
        fig.add_annotation(
            x='2023-11-01',
            y=110,
            text='First Forecast',
            showarrow=False
        )
        fig.update_xaxes(
            range=['2020-01-01', MSTL_forecast['ds'].max()]
        )
        return fig

    def seas_es_opt(self, train, h):
        with open('SeasonalExponentialSmoothingOptimized', 'rb') as model_file:
            SeasESOpt_model = joblib.load(model_file)

        SeasESOpt_forecast = SeasESOpt_model.predict(h=85 + h)

        fig = go.Figure()
        fig.add_trace(go.Scatter(x=train['ds'], y=train['y'], mode='lines', name='Dados de Treinamento (1987-05-20 - 2023-11-01)', line=dict(color='rgba(0, 255, 0, 0.8)')))
        fig.add_trace(go.Scatter(x=SeasESOpt_forecast['ds'], y=SeasESOpt_forecast['SeasESOpt'], mode='lines', name='Previsão SeasESOpt', line=dict(color='rgba(0, 255, 255, 0.8)')))
        fig.update_layout(
            xaxis_title='Data',
            legend=dict(x=0, y=1),
            xaxis=dict(tickformat='%Y-%m-%d')
        )
        fig.add_shape(
            type='line',
            x0='2023-11-01',
            x1='2023-11-01',
            y0=10,
            y1=140,
            line=dict(color='rgba(255, 255, 0, 0.8)', dash='dash')
        )
        fig.add_annotation(
            x='2023-11-01',
            y=100,
            text='First Forecast',
            showarrow=False
        )
        fig.update_xaxes(
            range=['2020-01-01', SeasESOpt_forecast['ds'].max()]
        )

        return fig