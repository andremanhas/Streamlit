import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
from prophet import Prophet
from sklearn.metrics import mean_absolute_error, mean_squared_error
import numpy as np
import joblib

# Função para calcular métricas
def calculate_metrics(y_true, y_pred):
    mae = mean_absolute_error(y_true, y_pred)
    mse = mean_squared_error(y_true, y_pred)
    mape = np.mean(np.abs((y_true - y_pred) / y_true)) * 100
    return mae, mse, mape

# Título da aplicação
st.title("Previsão de Preços de Petróleo - Prophet 2")

# Upload do dataset
uploaded_file = st.file_uploader("Faça upload de um arquivo Excel (.xlsx)", type="xlsx")

if uploaded_file is not None:
    # Leitura do dataset
    series = pd.read_excel(uploaded_file, engine='openpyxl')

    # Processamento dos dados
    series['data'] = pd.to_datetime(series['data'], format='%d/%m/%Y')
    series.set_index('data', inplace=True)
    series["preco_dia_anterior"] = series["preco"].shift(1)
    series = series.dropna()

    # Divisão entre treino e teste
    train_size = series.shape[0] - 14
    train = series[:train_size].reset_index()
    test = series[train_size:].reset_index()

    train_prophet = train.rename(columns={"data": "ds", "preco": "y"})
    test_prophet = test.rename(columns={"data": "ds", "preco": "y"})

    # Modelo Prophet 2
    model_prophet2 = Prophet(daily_seasonality=True)
    model_prophet2.add_regressor("preco_dia_anterior")

    # Treinando o modelo
    with st.spinner("Treinando o modelo Prophet 2..."):
        model_prophet2.fit(train_prophet)

    # Previsões
    future = test_prophet[["ds", "preco_dia_anterior"]]
    
    # Adiciona o próximo dia para prever
    next_day = pd.DataFrame({
        "ds": [future["ds"].iloc[-1] + pd.Timedelta(days=1)],
        "preco_dia_anterior": [series["preco"].iloc[-1]]
    })
    future = pd.concat([future, next_day], ignore_index=True)

    forecast = model_prophet2.predict(future)

    # Métricas de avaliação (para dados reais)
    preds = forecast[["ds", "yhat"]].iloc[:-1].set_index("ds")
    y_test = test_prophet.set_index("ds")["y"]
    metrics = calculate_metrics(y_test, preds["yhat"])

    # Exibição dos resultados
    st.subheader("Métricas de Avaliação")
    st.write(f"**MAE:** {metrics[0]:.2f}")
    st.write(f"**MSE:** {metrics[1]:.2f}")
    st.write(f"**MAPE:** {metrics[2]:.2f}")

    # Gráfico de comparação
    st.subheader("Gráfico de Previsões")
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(y_test.index, y_test.values, label="Valores Reais", color="blue")
    ax.plot(preds.index, preds["yhat"], label="Previsões", color="red", linestyle="--")
    ax.set_title("Valores Reais vs Previsões - Prophet 2")
    ax.set_xlabel("Data")
    ax.set_ylabel("Preço")
    ax.legend()
    st.pyplot(fig)

    # Previsão do próximo dia
    next_day_forecast = forecast.iloc[-1]
    st.subheader("Previsão para o Próximo Dia")
    st.write(f"Data: **{next_day_forecast['ds'].date()}**")
    st.write(f"Preço Previsto: **{next_day_forecast['yhat']:.2f}**")

    # Salvar o modelo treinado
    if st.button("Salvar Modelo"):
        joblib.dump(model_prophet2, "prophet2.joblib")
        st.success("Modelo salvo como 'prophet2.joblib'.")

# Mensagem padrão
else:
    st.info("Por favor, faça upload de um arquivo Excel com os dados para continuar.")
