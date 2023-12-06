from flask import Flask, render_template, request
from unidecode import unidecode
import pandas as pd
import pickle

app = Flask(__name__)

# Carregar o modelo CatBoost
with open('meu_projeto/srv/modelo_catboost.pkl', 'rb') as file:
    model = pickle.load(file)

# Função para pré-processar strings
def preprocess_strings(value):
    if isinstance(value, str):
        return unidecode(value.lower())
    return value

# Rotas
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    try:
        # Obter dados do formulário
        nome_ies = request.form['nome_ies']
        modalidade_ensino = request.form['modalidade_ensino']
        nome_curso = request.form['nome_curso']
        nome_turno_curso = request.form['nome_turno_curso']
        sexo_beneficiario = request.form['sexo_beneficiario']
        raca_beneficiario = request.form['raca_beneficiario']
        regiao_beneficiario = request.form['regiao_beneficiario']
        sigla_uf_beneficiario = request.form['sigla_uf_beneficiario']
        municipio_beneficiario = request.form['municipio_beneficiario']

        # Criar DataFrame com os dados do formulário
        dados_formulario = pd.DataFrame({
            'NOME_IES_BOLSA': [nome_ies],
            'MODALIDADE_ENSINO_BOLSA': [modalidade_ensino],
            'NOME_CURSO_BOLSA': [nome_curso],
            'NOME_TURNO_CURSO_BOLSA': [nome_turno_curso],
            'SEXO_BENEFICIARIO_BOLSA': [sexo_beneficiario],
            'RACA_BENEFICIARIO_BOLSA': [raca_beneficiario],
            'REGIAO_BENEFICIARIO_BOLSA': [regiao_beneficiario],
            'SIGLA_UF_BENEFICIARIO_BOLSA': [sigla_uf_beneficiario],
            'MUNICIPIO_BENEFICIARIO_BOLSA': [municipio_beneficiario]
        })

        # Aplicar pré-processamento aos dados de entrada
        for column in dados_formulario.columns:
            dados_formulario[column] = dados_formulario[column].apply(preprocess_strings)

        # Fazer previsão usando o modelo CatBoost
        predicao = model.predict(dados_formulario)

        # Exibir o resultado
        return render_template('result.html', predicao=predicao[0])

    except Exception as e:
        print("Erro ao processar a solicitação:", str(e))
        return render_template('error.html', error=str(e))

if __name__ == '__main__':
    app.run(debug=True)


