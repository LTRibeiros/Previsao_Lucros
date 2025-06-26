import pandas as pd
import numpy as np
from sklearn.linear_model import LinearRegression
from flask import Flask, render_template, request, jsonify

app = Flask(__name__)

@app.route('/')
def hello_world():
    return render_template('index.html')

@app.route('/enviar', methods=['POST'])
def enviar():
    file = request.files['csvFile']
    df = pd.read_csv(file)
    print(df)

    df['Mês'] = df['Mês'].astype(int)
    df['Lucro'] = df['Lucro'].astype(float)

    x = df['Mês'].values.reshape(-1, 1)
    y = df['Lucro'].values

    modelo = LinearRegression()
    modelo.fit(x, y)

    proximo_mes = x.max() + 1
    previsao = round(modelo.predict(np.array([[proximo_mes]]))[0])

    print(previsao)



    return render_template('previsao.html', proximo_mes=proximo_mes, previsao=previsao)

if __name__ == '__main__':
    app.run(Debug=True)
