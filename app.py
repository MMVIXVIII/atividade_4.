import streamlit as st
import tensorflow as tf
import numpy as np


# Título da Aplicação
st.title("Previsor de Horas de Estudo para o ENEM 🎯")
st.write("Um modelo simples com TensorFlow para estimar o esforço necessário.")

# 1. Dicionário de Dados (Tema ENEM: Nota vs Média de Horas de Estudo Semanais)
# Notas reais simuladas e as horas que os estudantes dedicaram
notas = np.array([500.0, 600.0, 700.0, 800.0, 900.0, 950.0], dtype=float)
horas_estudo = np.array([5.0, 10.0, 18.0, 25.0, 35.0, 42.0], dtype=float)

# 2. Criando e Treinando o Modelo Simples
modelo = tf.keras.Sequential([tf.keras.layers.Dense(units=1, input_shape=[1])])
modelo.compile(optimizer='sgd', loss='mean_squared_error')

# Treinamento rápido em segundo plano
with st.spinner("Treinando a rede neural..."):
    modelo.fit(notas, horas_estudo, epochs=500, verbose=0)

# 3. Interface Interativa
st.subheader("Simule sua Meta")
nota_desejada = st.slider("Qual nota você quer tirar no ENEM?", 500.0, 1000.0, 850.0, 10.0)

# Fazendo a previsão
horas_previstas = modelo.predict(np.array([nota_desejada]))[0][0]

# Exibindo o resultado
st.success(f"Para tirar **{nota_desejada}** pontos, a IA estima que você precisará estudar cerca de **{max(0, horas_previstas):.1f} horas semanais**.")



import numpy as np
import tensorflow as tf

# Dados realistas: [Horas de estudo diárias] e [Nota proporcional]
horas_estudo = np.array([1.0, 2.0, 3.0, 4.0, 5.0], dtype=float)
notas_enem = np.array([5.0, 6.5, 7.8, 8.8, 9.5], dtype=float)

# Criando a rede neural com um único neurônio
modelo = tf.keras.Sequential([
    tf.keras.layers.Dense(units=1, input_shape=[1])
])

# Compilando o modelo
modelo.compile(optimizer='sgd', loss='mean_squared_error')

# Treinando o modelo
modelo.fit(horas_estudo, notas_enem, epochs=500, verbose=0)

# Calculando as horas diárias necessárias para atingir a nota 10
peso, vies = modelo.layers[0].get_weights()
horas_para_dez = (10.0 - vies[0]) / peso[0]

print(f"Previsão: Para alcançar a nota 10, o modelo estima cerca de {horas_para_dez[0]:.2f} horas diárias de estudo.")