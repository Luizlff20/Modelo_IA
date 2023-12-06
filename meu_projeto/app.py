import numpy as np
import os
from flask import Flask, request, render_template, make_response
import joblib
import pandas as pd
from unidecode import unidecode  # Certifique-se de ter a biblioteca instalada: pip install unidecode

app = Flask(__name__, static_url_path='/static')
model = joblib.load('/content/modelo_catboost.pkl')

DEFAULT_PORT = 5500

# Função para pré-processar strings
def preprocess_strings(value):
    if isinstance(value, str):
        return unidecode(value.lower())
    return value

@app.route('/')
def display_gui():
    return render_template('template.html')

@app.route('/verificar', methods=['POST'])
def verificar():
    try:
        NOME_IES_BOLSA = request.form.get('universidade', '')
        MODALIDADE_ENSINO_BOLSA = request.form.get('grindRadionsModalidade', '')
        NOME_CURSO_BOLSA = request.form.get('curso', '')
        NOME_TURNO_CURSO_BOLSA = request.form.get('grindRadionsTurno', '')
        SEXO_BENEFICIARIO_BOLSA = request.form.get('grindRadionsSexo', '')
        RACA_BENEFICIARIO_BOLSA = request.form.get('grindRadionsCor', '')
        REGIAO_BENEFICIARIO_BOLSA = request.form.get('regiao', '')
        SIGLA_UF_BENEFICIARIO_BOLSA = request.form.get('uf', '')
        MUNICIPIO_BENEFICIARIO_BOLSA = request.form.get('municipio', '')

        # Aplicar pré-processamento aos dados de entrada
        dados_teste = pd.DataFrame({
            'NOME_IES_BOLSA': [NOME_IES_BOLSA],
            'MODALIDADE_ENSINO_BOLSA': [MODALIDADE_ENSINO_BOLSA],
            'NOME_CURSO_BOLSA': [NOME_CURSO_BOLSA],
            'NOME_TURNO_CURSO_BOLSA': [NOME_TURNO_CURSO_BOLSA],
            'SEXO_BENEFICIARIO_BOLSA': [SEXO_BENEFICIARIO_BOLSA],
            'RACA_BENEFICIARIO_BOLSA': [RACA_BENEFICIARIO_BOLSA],
            'REGIAO_BENEFICIARIO_BOLSA': [REGIAO_BENEFICIARIO_BOLSA],
            'SIGLA_UF_BENEFICIARIO_BOLSA': [SIGLA_UF_BENEFICIARIO_BOLSA],
            'MUNICIPIO_BENEFICIARIO_BOLSA': [MUNICIPIO_BENEFICIARIO_BOLSA],
        })

        # Aplicar pré-processamento aos dados de entrada
        for column in dados_teste.columns:
            dados_teste[column] = dados_teste[column].apply(preprocess_strings)

        # Fazer previsão usando o modelo treinado
        classe_predita = model.predict(dados_teste)[0]

        print(":::::::::: Dados de Teste :::::::::::")
        print("Sexo: {}".format(SEXO_BENEFICIARIO_BOLSA))
        print("Cor: {}".format(RACA_BENEFICIARIO_BOLSA))
        print("Nome da Universidade: {}".format(NOME_IES_BOLSA))
        print("Qual curso: {}".format(NOME_CURSO_BOLSA))
        print("Modalidade: {}".format(MODALIDADE_ENSINO_BOLSA))
        print("Horário: {}".format(NOME_TURNO_CURSO_BOLSA))
        print("Região: {}".format(REGIAO_BENEFICIARIO_BOLSA))
        print("UF: {}".format(SIGLA_UF_BENEFICIARIO_BOLSA))
        print("Município: {}".format(MUNICIPIO_BENEFICIARIO_BOLSA))
        print("Classe Predita: {}".format(str(classe_predita)))

    except Exception as e:
        print("Erro ao processar a solicitação:", str(e))
        classe_predita = "Erro"

    return render_template('template.html', classe=str(classe_predita))

if __name__ == "__main__":
    port = int(os.environ.get('PORT', DEFAULT_PORT))
    app.run(host='0.0.0.0', port=port)

