from llama_index.llms.groq import Groq
from llama_index.core import Settings
from llama_index.experimental.query_engine import PandasQueryEngine
import pandas as pd
import numpy as np
from llama_index.core.query_pipeline import (QueryPipeline as QP, Link, InputComponent)
from llama_index.core import PromptTemplate
from llama_index.experimental.query_engine.pandas import PandasInstructionParser
import textwrap
from llama_index.core.tools import FunctionTool
from llama_index.core.query_engine import CustomQueryEngine
import matplotlib.pyplot as plt
import logging

my_key = "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"
df = pd.read_csv("../data/fat_fat_id_agrup.csv", parse_dates=['MES_ANO']) # subtituir o local do arquivo depois de subir

# USEFULL FUNCTIONS

def descricao_colunas(df):
    descricao = '\n'.join([f"`{col}`: {str(df[col].dtype)}" for col in df.columns])
    return 'Aqui estão os detalhes das colunas do DataFrame:\n' + descricao

def formatar_texto(response):
    texto = response.message.content
    #texto = texto.split('\n\n')[0]
    texto_formatado = textwrap.fill(texto, width=100)

    return texto_formatado

def off_clients():
    all_clients = df['CLIENTE'].unique()
    now_clients = list(df[df['MES_ANO'] > '2022-12']['CLIENTE'].unique())
    off_clients = np.setdiff1d(all_clients, now_clients).tolist()

    return off_clients

def format_response(response):
    final_resp = str(response).split('assistant:')[1]
    return final_resp

# MODEL INSTRUCTIONS

instruction_str = (
    "1. Cada linha da tabela 'df' representa uma compra feita por um cliente em uma determinada data.\n"
    "2. Um mesmo cliente pode aparecer diversas vezes na tabela 'df' e cada linha representará uma de suas compras.\n"
    "3. A coluna 'CLIENTE' representa o código de identificação do cliente. Esse código se chama 'código de matriz'.\n"
    "4. A coluna 'FILIAL_CODIGO' apresenta os códigos das filiais da empresa. Esse código é chamado de 'código do canal de venda'.\n"
    "5. A coluna 'MES_ANO' representa o ano e o mês em que o cliente fez a compra e está no formato 'YYYY-mm-dd'.\n"
    "6. A coluna 'VALOR_MOVIMENTO' fornece o valor em dinheiro gasto pelo cliente na compra.\n"
    "7. A coluna 'QUANTIDADE' fornece a quantidade de produto comprada pelo cliente.\n"
    "8. A coluna 'SAFRA' fornece o código da safra em que a compra foi feita.\n"
    "9. Cada safra tem duração de 6 meses.\n"
    "10. O código da safra, apesar de não estar no formado de data, representa um período de tempo. Cada código representa um período de 6 meses."
    "10. A coluna 'ITEM' fornece o nome do item comprado pelo cliente.\n"
    "11. A coluna 'MARCA' fornece o nome da marca do item comprado pelo cliente.\n"
    "12. A coluna 'CODIGO_ITEM' fornece o código de identificação do item.\n"
    "13. Converta a consulta para código Python executável usando Pandas.\n"
    "14. Se a pergunta envolver maior ou menor valor de compra, não procure o maior ou menor valor no dataframe antes de agrupá-lo por cliente .\n"
    "15. Se a pergunta envolver maior ou menor quantidade de compra, não procure a maior ou menor quantidade no dataframe antes de agrupá-lo por cliente .\n"
    "16. Se a pergunta não especificar valor ou quantidade de compra, responder com as duas quantidades.\n"
    "17. Não usar 'idxmin()' ou 'idxmax()' sem agrupar e somar quantidade ou valor movimento.\n"
    "18. Quando perguntando se algum cliente compra conosco atualmente, olhar apenas registros a partir do ano de 2023.\n"
    "19. Quando perguntado quais clientes não compram a mais de duas safras, chamar a função 'off_clients()' e retornar para o usuário seu resultado.\n"
    "20. Responda sempre em português do Brasil.\n"
    "21. A linha final do código deve ser uma expressão Python que possa ser chamada com a função 'eval()'.\n"
    "22. O código deve representar uma solução para a consulta.\n"
    "23. IMPRIMA APENAS A EXPRESSÃO.\n"
    "24. Não coloque a expressão entre aspas.\n"
)

# Prompt que será enviado ao modelo para que ele gere o código desejado
pandas_prompt_str = (
    "Você está trabalhando com um dataframe do pandas em Python chamado 'df'.\n"
    "O dataframe 'df' possui informações de compras de clientes.\n"
    "Esse é o resultado de 'print(df.head())':\n"
    "{df_str}\n\n"
    "Siga essas instruções:\n"
    "{instruction_str}\n"
    "Consulta: {query_str}\n\n"
    "Expressão:"
)

# Prompt para guiar o modelo a sintetizar uma resposta com base nos resultados obtidos
response_synthesis_prompt_str = (
    "Dada uma pergunta de entrada, atue como analista de dados elabore uma resposta a partir dos resultados da consulta.\n"
    "Responda de forma natural, sem introduções como 'A resposta é:' ou algo semelhante.\n"
    "Consulta: {query_str}\n\n"
    "Instruções do Pandas (opcional):\n{pandas_instructions}\n\n"
    "Saída do Pandas: {pandas_output}\n\n"
)

pandas_prompt = PromptTemplate(pandas_prompt_str).partial_format(
    instruction_str=instruction_str, colunas_detalhes=descricao_colunas(df), df_str=df.head(5)
)
pandas_output_parser = PandasInstructionParser(df)
response_synthesis_prompt = PromptTemplate(response_synthesis_prompt_str)
llm = Groq(model="llama3-70b-8192", api_key=my_key)


# PIPELINE

query_pipe = QP(
    modules = {
        "input":InputComponent(),
        "pandas_prompt":pandas_prompt, 
        "llm1":llm,
        "pandas_output_parser":pandas_output_parser,
        "response_synthesis_prompt":response_synthesis_prompt,
        "llm2":llm
    },
    verbose=False
)

query_pipe.add_chain(['input', 'pandas_prompt', 'llm1', 'pandas_output_parser'])
query_pipe.add_links(
    [
        Link("input", "response_synthesis_prompt", dest_key='query_str'), 
        Link("llm1", "response_synthesis_prompt", dest_key='pandas_instructions'),
        Link("pandas_output_parser", "response_synthesis_prompt", dest_key='pandas_output')
    ]
)
query_pipe.add_link("response_synthesis_prompt", "llm2")

# DEFINING THE ANSWERS

str_senha_ex = "É provável que sua senha tenha expirado pois, de acordo com nossa política de segurança, elas são resetadas a cada 90 dias. Para atualizar sua senha, utilize o link abaixo. Lembre-se: a senha do seu e-mail é igual a do Conecta! " \
" - https://passwordreset.microsoftonline.com/ " \
" Caso precise de suporte do time de T.I, estamos disponíveis através dos canais:" \
" - Telefone: (43)3377-8646 " \
" - Sistema de chamados: https://belagricola.mysupport.net.br/"  

sem_acesso_filial = "Se você está com problemas para listar a filial, utilize o link para abrir um chamado:" \
" - https://belagricola.sharepoint.com/sites/Intranet/SitePages/link-externo.aspx?link=https%3A%2F%2Fbelagricola.sharepoint.com%2Fsuprimentos%2FLists%2FChamado+Acesso+Conecta%2FAllItems.aspx&titulo=Conecta+Chamados&OR=Teams-HL&CT=1650982127372&params=eyJBcHBOYW1lIjoiVGVhbXMtRGVza3RvcCIsIkFwcFZlcnNpb24iOiIyNy8yMjA0MDExMTQwOSJ9" \
"" \
" Caso precise de suporte do time de TI, estamos disponíveis através do telefone (43)3377-8646."

sem_safra = "Se você não consegue acessar a safra na lista de preços abra um chamado através do link:" \
" - LINK " \
" Caso precise de suporte do time de TI, estamos disponíveis através do telefone (43)3377-8646."

sem_reposicao = "Caso você esteja com problemas referentes a materiais sem reposição, abra um chamado através do link:" \
" - https://belagricola.sharepoint.com/sites/Intranet/SitePages/link-externo.aspx?link=https://belagricola.sharepoint.com/suprimentos/Lists/material_reposicao/AllItems.aspx&titulo=Material%20sem%20reposi%C3%A7%C3%A3o" \
"" \
" Caso precise de suporte do time de TI, estamos disponíveis através do telefone (43)3377-8646."

prod_similaridade = "Caso você esteja com problemas referentes à similaridade de produtos, abra um chamado através do link:" \
" - https://belagricola.sharepoint.com/sites/Intranet/SitePages/link-externo.aspx?link=https://belagricola.sharepoint.com/insumos/Lists/Similaridade/AllItems.aspx&titulo=Similaridade" \
" Caso precise de suporte do time de TI, estamos disponíveis através do telefone (43)3377-8646."

atualizar_listas = "Para atualizar as listas, por favor, siga os passos indicados no vídeo:" \
" - VIDEO" \
" Caso precise de suporte do time de TI, estamos disponíveis através do telefone (43)3377-8646."


# DICTIONARY Q&A

qa_dict = {
            "senha expirada": str_senha_ex,
            "sem acesso filial": sem_acesso_filial,
            "sem safra": sem_safra,
            "produto sem reposição": sem_reposicao,
            "produto similaridade": prod_similaridade,
            "atualizar listas": atualizar_listas 
        }


logging.getLogger().setLevel(logging.WARNING)
def query(query_str):
    if query_str == "me ajude a vender mais":
        response = query_pipe.run(query_str=query_str)
        return format_response(response)
    elif np.isin(query_str, list(qa_dict.keys())) == True:
        return qa_dict[query_str]
    else:
        response = query_pipe.run(query_str=query_str)
        return format_response(response)