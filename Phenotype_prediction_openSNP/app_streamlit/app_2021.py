# to run: streamlit run app.py
from matplotlib.pyplot import title
import streamlit as st
import pickle
import pandas as pd
import plotly.express as px
import numpy as np

st.title("Aplicativo Eye Color Prediction")
st.subheader("Este aplicativo tem como objetivo prever a cor dos olhos de uma pessoa usando apenas o seu material genético, utilizando Aprendizado de Máquina.")
st.write("")


st.write("Para isso serão utilizadas os dados referentes a variações em uma única base no DNA, chamadas SNPs. O valor de cada SNP pode ser encontrado a partir do processamento do material genético.")
st.markdown("> OBS: O modelo não funciona sem o valor para a SNP rs12913832.")
st.markdown("> OBS: Não existem dados faltantes para a SNP rs1393350, observados nos dados obtidos.")
# st.write("Página do projeto completo: ")
st.header("Selecione os dados da amostra a ser classificada:")


col1, col2, col3, col4, col5, col6 = st.columns(6)
with col1:
   rs12913832 = st.radio("rs12913832", ["AA","AG", "GG"])
with col2:
    rs1800407 = st.radio("rs1800407", ["CC","TC", "TT", "missing"])
with col3:
    rs12896399 = st.radio("rs12896399", ["GG","TG", "TT", "missing"])
with col4:
    rs16891982 = st.radio("rs16891982", ["CC","GC", "GG", "missing"])
with col5:
    rs1393350 = st.radio("rs1393350", ["GG", "GA", "AA"])
with col6:
    rs12203592 = st.radio("rs12203592", ["CC","TC", "CC", "missing"])


x_new = pd.DataFrame({"rs12913832":[rs12913832],
                     "rs1800407":[rs1800407],
                     "rs12896399":[rs12896399],
                     "rs16891982":[rs16891982],
                     "rs1393350":[rs1393350],
                    "rs12203592":[rs12203592]})


st.subheader("Dados de Entrada:")
st.table(x_new)

st.header("Resultado")

# Predição
filename = "best_model.sav"
model = pickle.load(open(filename, 'rb'))
y_predict = model.predict_proba(x_new)

# Exposição do resultado
categorias = ["Azul/Verde/Cinza","Castanho/Escuro","Intermediário"]

y = np.round(y_predict[0]*100, 2)
text = [f"{val:.2%}" for val in y_predict[0]]

fig = px.bar(y=categorias, x=y_predict[0], orientation='h', text=text, width=800,
     height= 600, title="Predição da Cor dos Olhos do Indivíduo")

fig.update_traces(textfont_size=20)

fig.update_layout(
    title_font_family = "Verdana",
    title_font_size = 22,
    xaxis = dict(
        tickmode = 'array',
        range=[0, 1],
        tickvals  = [0, 0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8, 0.9, 1],
        ticktext  = ["0%", "10%","20%","30%","40%","50%","60%","70%","80%","90%","100%"],
        title = "Probabilidade"
    ),
    yaxis = dict(
        tickfont = dict(size=14, family = "Verdana"),
        title = ""
    )
)


st.plotly_chart(fig)

for i in range(3):
    st.write(f"Probabilidade predita para cor {categorias[i]} = {y_predict[0][i]:.2%}")