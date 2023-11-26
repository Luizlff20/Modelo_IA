from flask import Flask, render_template, request
import pandas as pd
import pickle

app = Flask(__name__)

# Carregar o modelo CatBoost
with open('meu_projeto\\srv\\modelo_catboost.pkl', 'rb') as file:
    model = pickle.load(file)

# Rotas
@app.route('/')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
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

    # Fazer previsão usando o modelo CatBoost
    predicao = model.predict(dados_formulario)

    # Exibir o resultado
    return render_template('result.html', predicao=predicao[0])

if __name__ == '__main__':
    app.run(debug=True)
