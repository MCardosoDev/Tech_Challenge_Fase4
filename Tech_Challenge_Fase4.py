import streamlit as st
from utils import (
    DataExtractor,
    Loader,
    TimeSeriesPlotter,
    FillNulls,
    Adfuller,
    Stationary,
    Differentiation,
    Preparation,
    Models,
    BestModel,
    FitModels
)

def main():
    st.set_page_config(layout="wide")
    style = {'width': '100%'}

    st.sidebar.image('brent2.png', width=50)
    st.sidebar.title('Petróleo Brent')

    user_menu = st.sidebar.radio(
        'Painéis',
        (
            'Geral',
            'Análise Exploratória - EDA',
            'Análise da Série Temporal',
            'Transformação da Série',
            'Análise de Autocorrelação',
            'Avaliação dos Modelos',
            'Melhores Modelos',
            'Resultados'
        )
    )

    load = Loader()
    extract = DataExtractor()
    plotter = TimeSeriesPlotter()
    fill_nulls = FillNulls()
    adf = Adfuller()
    stationary = Stationary()
    differentiation = Differentiation()
    prep = Preparation()
    best_model = BestModel()
    model = Models()
    fit = FitModels()
    extract.extract_data()
    dados = load.load_data()
    dados_tans = load.load_data_trans()
    dados_dif = load.load_data_diff()
    df = fill_nulls.rolling_mean(dados_tans)
    
    while df.isnull().any().any():
        for column in df.columns:
            if df[column].isnull().any():
                df = fill_nulls.rolling_mean(df)

    dados_tans = load.load_data_trans()
    prep.data_prep(dados_tans, dados_dif)

    train = load.load_data_train()
    test = load.load_data_test()
    valid = load.load_data_valid()

    fit.fit(train)
    
    if user_menu == 'Geral':
        pass
        st.markdown("<h3 style='text-align:center; width: 100%;'>Análise de Preço por Barril do Petróleo Bruto Brent (FOB)</h3>", unsafe_allow_html=True)
        st.markdown('***')
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                            O desafio proposto é, analisar os dados do preço por barril do petróleo bruto Brent (FOB), 
                            desenvolver um dashboard interativo contendo insights que influenciaram alterações e variações dos preços, 
                            seguir um storytelling para situar o leitor e construir um modelo de machine learning para prever o preço do barril de petróleo diariamente (Série Temporal).
                            Para concluir, um planejamento para fazer o deploy do modelo em um ambiente de produção.
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )

        st.markdown('***')
        st.markdown("<h4 style='text-align:center; width: 100%;'>Dados Utilizados para Análise</h4>", unsafe_allow_html=True)
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                            <em>Análise dos dados históricos do IPEA</em> - Os dados sobre o histórico do preço por barril do petróleo bruto Brent (FOB) fornecido diariamente 
                            no site retirados da fonte: Energy Information Administration (EIA) com os valores em US$
                            Os dados do preço por barril do petróleo bruto Brent FOB (Free On Board) é coletado diariamente na fonte: <a href='https://www.eia.gov/'>Energy Information Administration (EIA)</a>
                            com os valores em US$
                            <br><br>
                            - Link: <a href='http://www.ipeadata.gov.br/ExibeSerie.aspx?module=m&serid=1650971490&oper=view'>Os dados da série histórica de preço foram coletados no site da IPEA</a>
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )
        
        st.markdown('***')
        st.markdown("<h4 style='text-align:center; width: 100%;'>Dados Históricos</h4>", unsafe_allow_html=True)
        st.dataframe(dados.head(5).reset_index(drop=True), use_container_width=True)
        st.markdown(
            '<span class="small-font">Dados utilizados do período entre 1987 até 2023</span>',
            unsafe_allow_html=True
        )

        st.markdown('***')
        st.markdown("<h4 style='text-align:center; width: 100%;'>Diferenciação da série para conversão em uma série estacionária</h4>", unsafe_allow_html=True)
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                            A diferenciação é usada para transformar uma série temporal não estacionária em uma série estacionária. 
                            Ou seja, para <em>remoção de tendências, estabilização da variância</em> e eliminação de sazonalidade.
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )
        
        st.markdown('***')
        st.markdown("<h4 style='text-align:center; width: 100%;'>Função de Autocorrelação (ACF) e Função de Autocorrelação parcial (PACF)</h4>", unsafe_allow_html=True)
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                            A ACF mede a correlação entre os valores passados e os valores presentes de uma série temporal, 
                            levando em consideração todos os atrasos possíveis e identifica padrões de correlação entre observações em diferentes pontos temporais.
                            <br>
                            A PACF mede a correlação direta entre dois pontos temporais, controlando os efeitos dos atrasos intermediários.
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )

        st.markdown('***')
        st.markdown("<h4 style='text-align:center; width: 100%;'>Teste e Avaliação de modelos preditivos</h4>", unsafe_allow_html=True)
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                            <em>Avaliação dos modelos para séries temporais</em> - Para encontrar o melhor modelo preditivo, utiliza-se testes de erros wMAPE para avaliar a precisão de acerto de cada modelo.
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )

        st.markdown('***')
        st.markdown("<h4 style='text-align:center; width: 100%;'>Melhores Modelos</h4>", unsafe_allow_html=True)
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                            <em>Apresentação dos modelos com melhor desempenho</em> - Com a avaliação dos erros cometidos utilizando dados de treino e teste, juntamente com dados originais e matematicamente transformados.
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )

        st.markdown('***')
        st.markdown("<h4 style='text-align:center; width: 100%;'>Resultado</h4>", unsafe_allow_html=True)
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                            Apresentação do modelo com melhor desempenho para a previsão do preço do barril bruto de petróleo, descrição sobre os processos usados para o deploy do modelo em produção e considerações finais.
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )

    if user_menu == 'Análise Exploratória - EDA':
        st.markdown("<h3 style='text-align:center; width: 100%;'>Dados Históricos - IPEA (Instituto de Pesquisa Econômica Aplicada)</h3>", unsafe_allow_html=True)
        st.markdown('***')
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                            O preço por barril do petróleo bruto tipo Brent. Produzido no Mar do Norte (Europa), Brent é uma classe de petróleo bruto que serve como benchmark para o preço internacional de diferentes tipos de petróleo. 
                            Neste caso, é valorado no chamado preço FOB (free on board), que não inclui despesa de frete e seguro no preço.
                            <br><br>
                            Mais informações: <a href='https://www.eia.gov/dnav/pet/TblDefs/pet_pri_spt_tbldef2.asp'>IPEA</a>
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )

        st.dataframe(dados.head(10).reset_index(drop=True), use_container_width=True)

        st.markdown('***')
        col1, col2 = st.columns([1, 1])
        with col1:
            st.markdown("<h5 style='text-align:center; width: 100%;'>Atualizar com dados mais recentes</h5>", unsafe_allow_html=True)
            st.write("***")
            if st.button('Atualizar Dados'):
                extract.extract_data()
                dados = load.load_data()
                dados_tans = load.load_data_trans()
                dados_dif = load.load_data_diff()
                df = fill_nulls.rolling_mean(dados_tans)
                            
                while df.isnull().any().any():
                    for column in df.columns:
                        if df[column].isnull().any():
                            df = fill_nulls.rolling_mean(df)

                dados_tans = load.load_data_trans()
                prep.data_prep(dados_tans, dados_dif)
                train = load.load_data_train()
                test = load.load_data_test()
                valid = load.load_data_valid()
                fit.fit(train)

                st.success('Dados atualizados com sucesso.')
        
        with col2:
            st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                        Ao solicitar a atualização dos dados é feito uma leitura da pagina do IPEA e a extração dos dados com os mais recentes disponibilizados, e armazenados em um arquivo .parquet.
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )
            
            code1 = """
                def extract_data(self):
                    response = requests.get(self.url)
                    html_content = response.content
                    soup = BeautifulSoup(html_content, 'html.parser')
                    element_center = soup.find('body').find('form').find_all('center')
                    second_center = element_center[1]
                    lines = second_center.find_all('tr')
                    ...
                    df.to_parquet(filename, index=False)
                """

            st.code(code1, language='python')
        
        st.markdown('***')
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                        Para plotar os dados da série temporal é necessário conferir e alterar  os tipos dos dados para <em>datetime</em> e <em>float</em>.
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )

        st.plotly_chart(plotter.plot(dados_tans), style = style, use_container_width = True)
        
        st.markdown('***')
        st.plotly_chart(plotter.decompose(dados_tans), style = style, use_container_width=True)

        st.markdown('***')
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                        Ao analisar tanto a série temporal quanto a série decomposta, nota-se alguns pontos bem claros que devem ser investigados. 
                        Nota-se que algum evento influenciou um forte aumento e queda do preço em, 1990, 2001 e 2008. 
                        Ocorreu um aumento desproporcional e também uma forte queda no mesmo ano em 2014, 
                        houveram outras duas quedas em 2016 e 2020 com um grande aumento em 2022.
                        <br><br>
                        Analisando a série decomposta nota-se que, a linha de tendência segue a série apenas suavizando-a, sem dar mais informações. 
                        A série tem uma sazonalidade anual observada e pelo ruído nota-se que as maiores variações ocorreram entre 2008 e 2009 e em 2020.
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )

    if user_menu == 'Análise da Série Temporal':
        pass
        st.markdown("<h3 style='text-align:center; width: 100%;'>Análise da Série Temporal</h3>", unsafe_allow_html=True)
        st.markdown('***')
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                        Ao analisar novamente a série nota-se que impactos globais interferem nos exatos pontos de anomalia do dados.
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )
        st.plotly_chart(plotter.plot_with_events(dados_tans), style = style, use_container_width=True)
        
        st.markdown('***')
        st.markdown("""
            **Eventos Globais e seu Impacto nos Preços do Petróleo (FOB)**

            - **1990:** A Guerra do Golfo (Operação Tempestade no Deserto).
                - A Guerra do Golfo (Operação Tempestade no Deserto) começou em 1990, levando a um aumento da volatilidade nos preços do petróleo devido às preocupações com a oferta.
                A guerra entre "Kuwait" e Iraque fez com que o preço do petróleo despencasse. [Fonte](https://brasilescola.uol.com.br/historiag/guerra-golfo.htm)
            
            - **2001:** Ataques de 11 de setembro.
                - Os ataques de 11 de setembro de 2001 tiveram um impacto significativo nos mercados financeiros e nas economias globais, reduzindo o preço do petróleo e disparando diante da possibilidade de uma guerra entre os Estados Unidos e o Oriente Médio, gerando demanda em períodos de incertezas. [Fonte](https://www.suno.com.br/noticias/11-de-setembro-terror-mercados-bolsas/)
            
            - **2008:** Crise financeira global.
                - Em 2008, atingiu o ápice com o maior preço da história no barril. No primeiro semestre, uma soma de fatores levou os preços às alturas: tensões geopolíticas do Irã, Nigéria e Paquistão, o problema entre uma oferta limitada e uma demanda puxada pelos países emergentes, a conscientização de que as reservas são limitadas e de acesso cada vez mais difícil, uma febre dos fundos de investimento por matérias-primas. E a maior queda já registrada após o estouro da bolha imobiliária nos EUA levando o planeta a uma crise financeira global. Resultando em uma grande recessão global. A demanda por petróleo diminuiu drasticamente, levando a uma queda nos preços após um aumento inicial. [Fonte](https://g1.globo.com/Noticias/Economia_Negocios/0,,MUL940136-9356,00-O+ANO+EM+QUE+O+PETROLEO+ENLOUQUECEU+O+MERCADO.html)
            
            - **2014-2016:** Excesso de oferta global de petróleo.
                - Durante esse período, houve um excesso de oferta global de petróleo devido à produção elevada, especialmente por países membros da OPEP (Organização dos Países Exportadores de Petróleo). Isso contribuiu para a queda nos preços do petróleo. [Fonte](https://g1.globo.com/economia/noticia/2015/01/entenda-queda-do-preco-do-petroleo-e-seus-efeitos.html)
            
            - **2020:** Pandemia de COVID-19 e queda drástica na demanda.
                - A pandemia de COVID-19 teve um impacto dramático na demanda global por petróleo, resultando em uma queda acentuada nos preços. Restrições de viagem e lockdowns afetaram significativamente o consumo de petróleo. [Fonte](https://repositorio.ipea.gov.br/bitstream/11058/10213/1/bepi_27_covid19.pdf)
            
            - **2022:** Necessário verificar eventos específicos.
                - O aumento em 2022 pode ser explicado por uma recuperação econômica global pós-pandemia, aumento da demanda por petróleo e possíveis mudanças nas políticas globais de energia. E a guerra da Rússia-Ucrânia colaborou para o aumento do preço do petróleo, devido às preocupações de que as sanções, tendo a Rússia como alvo, prejudiquem o fornecimento de energia para o restante do mundo. [Fonte](https://www.cnnbrasil.com.br/economia/entenda-por-que-o-preco-do-petroleo-disparou-com-a-guerra-entre-ucrania-e-russia/)
        """)
        st.markdown('***')

        st.markdown("<h4 style='text-align:center; width: 100%;'>Ao analisar os dados de mais de uma semana nota-se muitos dados ausentes.</h4>", unsafe_allow_html=True)
        st.dataframe(dados.head(10).reset_index(drop=True), use_container_width=True)
        
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                            Com o dataset com esse padrão, gera-se um problema ao depender do modelo que for utilizado para realizar o forecast. Alguns modelos utilizam média móvel, janelas de tempo ou último dado disponível, são muitos tipos de modelos e dados faltantes geram um problema de ajuste e dificuldade em encontrar o melhor modelo para o caso.
                            <br><br>
                            Para ser ajustado esse problema, foi utilizado uma janela rolante com o cálculo de média móvel, como o método forward fill, mas não apenas com o último dia, mas uma média dos últimos 5 dias. E foram utilizados apenas dados anteriores aos dias com falta de dados para evitarmos o lookahead (informar dados futuros para o modelo).
                        </p>
                    """, 
                        unsafe_allow_html=True
                    )
        
        code = """
            date_range = pd.date_range(start=df.index.min(), end=df.index.max(), freq='D')
                df = df.reindex(date_range)
                rolling_mean = df[column_y].rolling(window=window_size, min_periods=1).mean()
                is_null_sequence = df[column_y].isnull()
                df[column_y] = np.where(is_null_sequence, rolling_mean, df[column_y])
                df[column_y] = df[column_y].round(2)
                df[column_ds] = df.index
                df.index.name = index
            """

        st.code(code, language='python')
        st.markdown('***')
        st.markdown("<h4 style='text-align:center; width: 100%;'>Ao analisar os dados agora nota-se que não se mais dados ausentes.</h4>", unsafe_allow_html=True)
        st.dataframe(dados_tans.tail(10), use_container_width=True)
        

    if user_menu == 'Transformação da Série':
        pass
        st.markdown("<h3 style='text-align:center; width: 100%;'>Transformação da Série</h3>", unsafe_allow_html=True)
        st.markdown('***')
        st.markdown("""
                        <p style='text-align: center; font-size: 18px;'>
                        A estacionariedade em séries temporais é essencial para garantir previsões precisas. Séries não estacionárias podem conter padrões temporais complexos, prejudicando a eficácia dos modelos. 
                        O Teste Augmented Dickey-Fuller (ADF) é uma ferramenta comum para verificar isso. Ele avalia se uma série possui raiz unitária, indicando não estacionariedade. 
                        Se o valor p do teste for inferior a um nível de significância (geralmente 0,05), rejeita-se a hipótese nula de não estacionariedade. 
                        A estacionariedade facilita a aplicação de modelos mais precisos, melhorando a qualidade das previsões.
                        </p>
                        """, 
                        unsafe_allow_html=True
                )

        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h5 style='text-align:center; width: 100%;'>Hipóteses do Teste ADF:</h5>", unsafe_allow_html=True)
            st.write("  - H0: A série não é estacionária.")
            st.write("  - H1: A série é estacionária.")
            st.write("***")
            if st.button('Executar Teste ADF'):
                with col2:
                    result = adf.test_adfuller(dados_tans['y'])

                    st.markdown("<h5 style='text-align:center; width: 100%;'>Teste ADF</h5>", unsafe_allow_html=True)
                    st.write(f'Teste estatístico: {result[0]}')
                    st.write(f'P-Value: {result[1]}')
                    st.write("***")
                    st.markdown("<h6 style='text-align:center; width: 100%;'>Valores críticos</h6>", unsafe_allow_html=True)
                    critical_values_table = {'Key': list(result[4].keys()), 'Value': list(result[4].values())} #type: ignore
                    st.table(critical_values_table)
                with col3:
                    st.markdown("<h5 style='text-align:center; width: 100%;'>Resultado</h5>", unsafe_allow_html=True)
                    st.write("***")
                    st.markdown("<span style='color:red'>O p-value de 27% e teste estatístico maior que os valores criticos comprova-se estatisticamente que não é estacionaria.</span>", unsafe_allow_html=True)

        st.write("***")

        st.markdown("<h5 style='text-align:center; width: 100%;'>Transformação para um Série Temporal estacionaria</h5>", unsafe_allow_html=True)
        
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Realizar transformações matemáticas em uma série temporal não estacionária para torná-la estacionária permite a aplicação de técnicas estatísticas e modelos de previsão com esse pressuposto.
                    Isso é fundamental para compreender padrões temporais e obter previsões mais precisas. As transformações realizadas na série foram:

                    - Aplicação do Logaritmo, para reduzir flutuações
                    - Subtração da Média Móvel com janela de tamanho 12
                    - Primeira diferenciação, para reduzir tendências

                    E a transformação Box-Cox é uma técnica estatística utilizada para estabilizar a variância e tornar os dados mais próximos de uma distribuição normal. 
                    Ela é frequentemente aplicada a conjuntos de dados que exibem heterocedasticidade (variância não constante) e não seguem uma distribuição normal.
                    A transformação Box-Cox é definida pela seguinte expressão:
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
        st.markdown(r"$$y^{(\lambda)} = \begin{cases} \frac{{y^\lambda - 1}}{{\lambda}}, & \text{se } \lambda \neq 0 \\ \log(y), & \text{se } \lambda = 0 \end{cases}$$")
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Calcular <em>λ</em> que maximiza a log-verossimilhança dos dados transformados e aplicar a transformação Box-Cox usando o valor estimado de <em>λ</em>.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
        
        data, data_mean, data_std = stationary.make_stationary(dados_tans)
        data_boxcox, data_mean_boxcox, data_std_boxcox, lambda_value = stationary.make_stationary_boxcox(dados_tans)

        st.plotly_chart(differentiation.data_diff(data, data_mean, data_std, "Diferenciação Logarítmica"), style = style, use_container_width = True)
        st.markdown("***")
        st.plotly_chart(differentiation.data_diff(data_boxcox, data_mean_boxcox, data_std_boxcox, "Box-Cox"), style = style, use_container_width = True)

        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Nota-se que com ambos os cálculos utilizados para realizar a transformação obtém um resultado muito semelhante.
                    Para esse caso segue com a utilização da transformação logarítmica.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Nota-se as mesmas anomalias observadas no gráfico do resíduo da decomposição no resultado da transformação matemática.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
        
        st.markdown("***")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.markdown("<h5 style='text-align:center; width: 100%;'>Novo Teste ADF:</h5>", unsafe_allow_html=True)
            st.write("  - H0: A série não é estacionária.")
            st.write("  - H1: A série é estacionária.")
            st.write("***")
            if st.button('Executar Novo Teste ADF'):
                with col2:
                    result = adf.test_adfuller(dados_dif['y'])

                    st.markdown("<h5 style='text-align:center; width: 100%;'>Teste ADF</h5>", unsafe_allow_html=True)
                    st.write(f'Teste estatístico: {result[0]}')
                    st.write(f'P-Value: {result[1]}')
                    st.write("***")
                    st.markdown("<h6 style='text-align:center; width: 100%;'>Valores críticos</h6>", unsafe_allow_html=True)
                    critical_values_table = {'Key': list(result[4].keys()), 'Value': list(result[4].values())} #type: ignore
                    st.table(critical_values_table)
                with col3:
                    st.markdown("<h5 style='text-align:center; width: 100%;'>Resultado</h5>", unsafe_allow_html=True)
                    st.write("***")
                    st.markdown("<span style='color:green'>P-value de 0.0% e teste estatístico menor que os valores críticos comprova-se estatisticamente que a série é estacionaria após a diferenciação.</span>", unsafe_allow_html=True)
        
    if user_menu == 'Análise de Autocorrelação':
        pass
        st.markdown("<h3 style='text-align:center; width: 100%;'>Autocorrelação (ACF) e Autocorrelação parcial (PACF)</h3>", unsafe_allow_html=True)
        st.markdown('***')
        st.markdown("""
                <p style='text-align: center; font-size: 18px;'>
                Os gráficos de autocorrelação (ACF) e autocorrelação parcial (PACF) são ferramentas essenciais na análise de séries temporais.
                ACF revela correlações em diferentes defasagens, enquanto PACF ajuda a identificar correlações diretas e eliminação de defasagens irrelevantes em modelos de previsão.
                Ao observar onde essas correlações se tornam não significativas nos gráficos PACF e ACF, você pode obter sugestões para os parâmetros Q (ordem do termo MA) e P (ordem do termo AR) para modelos como o ARMA, ARIMA ou SARIMA, os quais ajudam a modelar adequadamente a série temporal.

                - 5% ACF (intervalo de confiança).
                - 1.96/sqrt(N-d)
                    - *N* número de pontos e *d* é o número de vezes que os dados foram diferenciados (intervalo de confiança para estimativas de autocorrelação significativa).
                </p>
                """, unsafe_allow_html=True
            )
        
        st.markdown('***')
        st.write("### Série sem transformação matemática")
        st.pyplot(plotter.plot_acf_pacf(dados), use_container_width=True)

        st.markdown("""
                <p style='text-align: center; font-size: 18px;'>
                É notável que a série original, sem a aplicação das transformações matemáticas, apresenta uma forte autocorrelação entre seus valores. 
                Essa correlação diminui à medida que o número de lags aumenta. Por outro lado, a autocorrelação parcial mostra-se significativa apenas 
                para lags pequenos, uma vez que, para valores de lag maiores, a correlação parcial diminui consideravelmente.
                </p>
                """, unsafe_allow_html=True
            )
        st.markdown('***')
        st.markdown("<h5 style='text-align:center; width: 100%;'>Série com transformação matemática</h5>", unsafe_allow_html=True)
        st.plotly_chart(plotter.acf_pacf(dados_dif), style = style, use_container_width = True)
        
        st.markdown("""
                <p style='text-align: center; font-size: 18px;'>
                Nota-se que após a aplicação da transformação matemática para alcançar a estacionariedade, 
                os gráficos de ACF e PACF se tornam semelhantes. Isso sugere que a correlação direta 
                desempenha um papel predominante em comparação com a correlação indireta.
                </p>
                """, unsafe_allow_html=True
            )
        
        st.markdown('***')

        st.markdown("""
        <p style='text-align: left; font-size: 18px;'>
            Ordem de diferenciação *D* = 1 (Foi necessária 1 diferenciação para tornar a série estacionária)<br>
            *Q acf* = 0.85<br>
            *P pacf* = 0.85
        </p>
        """, unsafe_allow_html=True)

    if user_menu == 'Avaliação dos Modelos':
        pass
        st.markdown("<h3 style='text-align:center; width: 100%;'>Avaliação dos modelos - Métricas de Erro em Previsões</h3>", unsafe_allow_html=True)
        st.markdown('***')
        st.markdown("""
                <p style='text-align: center; font-size: 18px;'>
                    Foram utilizados uma série de modelos para realizar o forecast dos preços do barril de petróleo bruto que são amplamente usados para avaliação.
                    E utilizado a biblioteca `statsforecast` para gerar os modelos.
                    Modelos de séries temporais são estruturas analíticas que visam compreender e prever padrões em dados sequenciais ao longo do tempo, 
                    permitindo a captura de tendências, sazonalidades e variações temporais subjacentes.
                    Esses modelos são fundamentais para a tomada de decisões.
                </p>
                """, unsafe_allow_html=True
            )

        model_explanations = {
            "AutoETS":
                "O AutoETS (Automatic Exponential Smoothing) é um método automatizado que seleciona automaticamente o melhor modelo de suavização exponencial para uma série temporal. "
                "Ele considera diferentes configurações de suavização (além de sazonalidade e tendência) para melhor se ajustar aos padrões da série e produzir previsões precisas.",

            "Seasonal Window Average":
                "O método SeasonalWindowAverage calcula a média dos valores em uma janela móvel de tamanho fixo ao longo da série temporal, considerando uma sazonalidade específica. "
                "Ele suaviza flutuações sazonais de curto prazo, proporcionando uma visão da tendência sazonal.",

            "Dynamic Optimized Theta":
                "O Dynamic Optimized Theta otimiza dinamicamente os parâmetros de suavização Theta para se adaptar às características específicas da série temporal."
                "Como um ajuste personalizado que evolui conforme a série temporal se desenvolve, resultando em previsões mais ajustadas e eficientes.",

            "AutoTheta":
                "O AutoTheta é uma versão automatizada do método Theta, que é uma técnica de suavização exponencial dupla para séries temporais com sazonalidade aditiva.",

            "SeasESOpt (Seasonal Exponential Smoothing Optimized)":
                "O SESOt é uma variação do SES que otimiza automaticamente os parâmetros de suavização exponencial para melhor se ajustar aos dados da série temporal."
                "Essa abordagem busca ajustar dinamicamente os parâmetros para otimizar a previsão, garantindo uma resposta eficaz às variações específicas associadas à sazonalidade dos dados.",

            "MSTL":
                "O MSTL (Multiple Seasonal Decomposition of Time Series) é um método que decompõe séries temporais em diferentes componentes, como sazonalidade e tendência, "
                "para modelagem mais precisa. Ele suporta múltiplos comprimentos sazonais e utiliza o modelo AutoETS para previsões.",

            "AutoCES":
                "O AutoCES é um modelo avançado de Suavização Exponencial Complexa que seleciona automaticamente o melhor modelo usando um critério de informação, como o Critério de Informação de Akaike (AICc)." 
                "Os modelos específicos são estimados por meio da máxima verossimilhança." 
        }

        st.markdown('***')
        col1, col2 = st.columns([1, 2])
        with col1:
            selected_model = st.radio("Melhores modelos por desempenho", list(model_explanations.keys()))

        with col2:
            st.write('Modelos')
            st.write("***")
            st.write(model_explanations[selected_model]) # type: ignore

        st.markdown('***')

        st.markdown("<h5 style='text-align:center; width: 100%;'>Avaliação</h5>", unsafe_allow_html=True)
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Para avaliar os modelos foi levado em consideração a taxa de erro <em>wMAPE</em> entre os dados de teste e validação e os dados previstos.
                    O wMAPE pondera os erros relativos pelo valor real, o que significa que os erros em observações com valores maiores têm um impacto maior no resultado final. 
                    Essa métrica é especialmente útil quando os valores reais variam consideravelmente ao longo do tempo, e é importante dar mais importância a previsões precisas em momentos críticos.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
        st.markdown(r"$$WMAPE = \frac{\sum_{i=1}^{n} \left| y_{\text{true},i} - y_{\text{pred},i} \right|}{\sum_{i=1}^{n} \left| y_{\text{true},i} \right|} \times 100$$")
        st.markdown('***')

        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                        Ao avaliar o desempenho dos modelos, tanto com quanto sem a aplicação da transformação matemática, observa-se que os resultados 
                        foram consistentemente melhores quando treinados com a série original, sem a necessidade da transformação. Essa melhoria se deve, 
                        em grande parte, ao fato dos modelos avaliados serem capazes de lidar eficazmente com características intrínsecas, como 
                        tendência e sazonalidade, ou já incorporam mecanismos automáticos para tratar dessas características e o próprio modelo realiza a transformação da série.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
        
        st.markdown('***')
        st.markdown("<h5 style='text-align:center; width: 100%;'>Os períodos para treinar, testar e validar foram:</h5>", unsafe_allow_html=True)
        st.code("""
                latest_date = max(dataset['ds'])

                train_end_date = latest_date - timedelta(days=15)
                test_end_date = latest_date - timedelta(days=7)

                train = dataset.loc[dataset['ds'] < train_end_date]
                test = dataset.loc[(dataset['ds'] >= train_end_date) & (dataset['ds'] < test_end_date)]
                valid = dataset.loc[dataset['ds'] >= test_end_date]
                """
        )

        st.markdown('***')
        df_error = load.load_data_error()
        st.plotly_chart(plotter.plot_error(df_error), style = style, use_container_width=True)
        st.markdown('***')
        
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Observa-se com a analise de erros que os modelos <em>SeasESOpt (Seasonal Exponential Smoothing Optimized)</em> e o 
                    <em>MSTL (Multiple Seasonal Decomposition of Time Series)</em> obtém a melhor taxa de erro entre treino e validação.
                    E requerem uma analise mais aprofundada.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )

    if user_menu == 'Melhores Modelos':
        pass
        st.markdown("<h3 style='text-align:center; width: 100%;'>Modelos com melhores desempenhos</h3>", unsafe_allow_html=True)
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                        Após a avaliação dos erros no <em>wMAPE</em> gerados pelo modelo, observou-se que dois modelos obtiveram erros entre 3% e 5% o modelo 
                        <em>Multiple Seasonal Decomposition of Time Series - MSTL</em>  e o <em>SeasESOpt - (Seasonal Exponential Smoothing Optimized)</em> 
                        O MSTL (Decomposição Múltipla de Sazonalidade-Tendência usando LOESS) decompõe a série temporal em múltiplas sazonalidades (season_length=[180, 365]) 
                        usando regressão local LOESS (Locally Estimated Scatter Plot Smoothing).
                        Ajustando modelos de regressão localmente, em vez de globalmente, útil para capturar padrões não lineares em séries temporais e dados espaciais.
                        Ode suaviza o s pontos de dados que envolve uma janela móvel para estimar o valor suavizado de cada ponto com base em seus vizinhos.
                        Em seguida, faz previsões para a tendência usando um modelo personalizado não sazonal, e para cada sazonalidade, usa outro modelo selecionado.
                        Incrementado com o modelo AutoETS(model='AZN') Exponential Smoothing State Space Model (ETS) que ajusta automaticamente seus parâmetros 
                        ETS (Error, Trend, Seasonality) com base em um critério de informação, sendo o padrão o Critério de Informação de Akaike corrigido (AICc). 
                        Esse critério é uma medida que avalia a adequação do modelo e penaliza modelos mais complexos, favorecendo aqueles que ajustam bem os dados com menos parâmetros.
                        No caso específico de AZN, isso significa que o modelo está configurado para avaliar a adequação de componentes multiplicativas para erro e sazonalidade, 
                        e uma componente não sazonal para a tendência. O componente Z indica que a função deve otimizar automaticamente a escolha da forma das componentes.
                    </p>
                    """, 
                    unsafe_allow_html=True
                )
        
        st.code('''
                StatsForecast([MSTL(season_length=[180, 365], trend_forecaster=AutoETS(model='ZZN'))], freq='D', n_jobs=-1)
                ''')

        mstl = model.mstl(train, test, valid)
        st.write(f"WMAPE (Weighted Mean Absolute Percentage Error): {mstl[1]:.2%}")
        st.plotly_chart(mstl[0], style = style, use_container_width=True)

        st.write(f"WMAPE (Weighted Mean Absolute Percentage Error): {mstl[3]:.2%}")
        st.plotly_chart(mstl[2], style = style, use_container_width=True)
        
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                        O modelo SeasonalExponentialSmoothingOptimized é projetado para prever séries temporais usando uma média ponderada de todas as observações passadas, 
                        onde os pesos diminuem exponencialmente no passado. Este modelo é mais adequado para dados que não apresentam uma tendência ou sazonalidade clara.
                        A fórmula para a previsão de um passo à frente, assumindo que há t observações e uma sazonalidade de s períodos, é dada por:        
                    </br>
                    """, 
                    unsafe_allow_html=True
                )
        st.latex(r"\hat{y}_{t+1,s} = \alpha y_t + (1 - \alpha) \hat{y}_{t-1,s}")
        
        st.code('''
                SeasESOpt_model = StatsForecast(models=[SeasonalExponentialSmoothingOptimized(season_length=180)], freq='D', n_jobs=-1)
                ''')
        
        seas_es_opt = model.seas_es_opt(train, test, valid)
        st.write(f"WMAPE (Weighted Mean Absolute Percentage Error): {seas_es_opt[1]:.2%}")
        st.plotly_chart(seas_es_opt[0], style = style, use_container_width=True)

        st.write(f"WMAPE (Weighted Mean Absolute Percentage Error): {seas_es_opt[3]:.2%}")
        st.plotly_chart(seas_es_opt[2], style = style, use_container_width=True)

    if user_menu == 'Resultados':
        pass
        st.markdown("<h3 style='text-align:center; width: 100%;'>Resultado do Challenge</h3>", unsafe_allow_html=True)
        st.markdown('***')

        num_dates = st.slider('Número de Dias para Prever', min_value=1, max_value=180, value=7)
        best_model_mstl = best_model.mstl(train, test, valid, num_dates)
        st.plotly_chart(best_model_mstl, style = style, use_container_width=True)

        best_model = best_model.seas_es_opt(train, test, valid, num_dates)
        st.plotly_chart(best_model, style = style, use_container_width=True)

        st.markdown("***")
        
        st.markdown("<h4 style='text-align:center; width: 100%;'>Desenvolvimento do Modelo de Previsão do Preço do Barril de Petróleo Brent</h4>", unsafe_allow_html=True)

        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Durante o processo de desenvolvimento do modelo de previsão do preço do barril de petróleo Brent, adotamos
                    uma abordagem estruturada e eficiente, utilizando diversas ferramentas e boas práticas.
                    </br>
                    """, 
                    unsafe_allow_html=True
                )

        st.markdown("<h5 style='text-align:center; width: 100%;'>1. Escolha da Biblioteca e Modelagem</h5>", unsafe_allow_html=True)
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Optamos pela biblioteca Statsforecast para a construção de dois modelos de previsão. Essa escolha foi 
                    baseada no desempenho e eficácia dessa ferramenta nas previsões temporais dos dados apresentados.
                    </br>
                    """, 
                    unsafe_allow_html=True
                )

        st.markdown("<h5 style='text-align:center; width: 100%;'>2. Controle de Versão com GitHub</h5>", unsafe_allow_html=True)
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Para garantir o versionamento do código e facilitar a colaboração, utilizamos o GitHub. Essa plataforma
                    possibilitou um ambiente colaborativo, permitindo o acompanhamento das alterações e a fácil reversão em caso
                    de necessidade.
                    </br>
                    """, 
                    unsafe_allow_html=True
                )

        st.markdown("<h5 style='text-align:center; width: 100%;'>3. Implementação de Modelos Treinados</h5>", unsafe_allow_html=True)
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Utilizamos a biblioteca Joblib para transformar os modelos treinados em arquivos binários. Essa abordagem
                    facilita a integração dos modelos no código, proporcionando eficiência na manipulação e execução.
                    </br>
                    """, 
                    unsafe_allow_html=True
                )

        st.markdown("<h5 style='text-align:center; width: 100%;'>4. Disponibilização de Resultados com Streamlit Cloud</h5>", unsafe_allow_html=True)
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Para disponibilizar os resultados de maneira acessível, escolhemos o Streamlit Cloud. Essa plataforma oferece 
                    uma interface amigável e possibilita compartilhar visualizações interativas dos modelos de previsão.
                    </br>
                    """, 
                    unsafe_allow_html=True
                )

        st.markdown("<h5 style='text-align:center; width: 100%;'>5. Estruturação do Código em Classes</h5>", unsafe_allow_html=True)
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Convertemos todas as funções desenvolvidas durante o projeto em classes. Essa abordagem torna o código mais 
                    modular, facilitando a manutenção e a reutilização de componentes. Incluindo a automação da extração dos dados 
                    do site da IPEA.
                    </br>
                    """, 
                    unsafe_allow_html=True
                )

        st.markdown("<h5 style='text-align:center; width: 100%;'>6. Automação de Entradas de Dados e Armazenamento Eficiente</h5>", unsafe_allow_html=True)
        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Automatizamos o processo de entrada de dados e a separação de datasets, armazenando-os em formato Parquet.
                    Essa escolha visa otimizar o desempenho do sistema, garantindo eficiência no processamento de grandes conjuntos de dados.
                    </br>
                    """, 
                    unsafe_allow_html=True
                )

        st.markdown("""
                    <p style='text-align: center; font-size: 18px;'>
                    Com a aplicação dessas práticas e ferramentas, o trabalho se tornou mais escalável, fácil de manter e oferece 
                    resultados precisos na previsão do preço do barril de petróleo Brent. O deploy em produção foi realizado de 
                    forma eficiente, garantindo a acessibilidade e usabilidade dos modelos desenvolvidos.
                    </br>
                    """, 
                    unsafe_allow_html=True
                )
                
if __name__ == "__main__":
    main()