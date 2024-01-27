# Tech_Challenge_Fase4
### Análise dos dados do preço por barril do petróleo bruto Brent (FOB)

Fonte: Energy Information Administration (EIA)

Preço por barril do petróleo bruto tipo Brent. Produzido no Mar do Norte (Europa), Brent é uma classe de petróleo bruto que serve como benchmark para o preço internacional de diferentes tipos de petróleo. Neste caso, é valorado no chamado preço FOB (free on board), que não inclui despesa de frete e seguro no preço. Mais informações: https://www.eia.gov/dnav/pet/TblDefs/pet_pri_spt_tbldef2.asp

> Para visualizar online
>
>> <https://techchallengefase4.streamlit.app>
> 
> Para visualizar as análises em localhost rodar
>
>> **streamlit run Tech_Challenge_Fase4.py**
>

## Libs

- streamlit
- requests
- BeautifulSoup
- pandas
- numpy
- joblib
- scipy.stats 
  - boxcox
- datetime
  -  datetime
  -  timedelta
- plotly.express
- plotly.subplots
- plotly.graph_objs
- matplotlib.pyplot
- statsmodels.graphics.tsaplots
  - plot_acf
  - plot_pacf
- statsmodels.tsa.seasonal
  - seasonal_decompose
- statsmodels.tsa.stattools
  - adfuller
  - acf
  - pacf
- statsforecast
  - StatsForecast
- statsforecast.models
  - SeasonalExponentialSmoothingOptimized
  - MSTL
  - AutoETS