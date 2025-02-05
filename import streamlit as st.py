import streamlit as st
import json
from langchain.chains.sql_database.query import create_sql_query_chain
from langchain_community.agent_toolkits import create_sql_agent
from langchain_ollama.chat_models import ChatOllama
from langchain_community.utilities import SQLDatabase
from langchain_core.output_parsers import StrOutputParser
from langchain_core.prompts import ChatPromptTemplate
from langchain.agents import AgentType

# Configurar Streamlit
st.set_page_config(page_title="Chatbot de Geología", layout="wide")
st.title("Chatbot de Geología basado en Tesis")

# Cargar base de datos SQLite
url_query = "sqlite:///geodata1.db"  # Asegúrate de que el archivo está en el mismo directorio
db = SQLDatabase.from_uri(url_query, sample_rows_in_table_info=3)

# Inicializar el modelo LLM
llm = ChatOllama(model='llama3:latest', temperature=0.5)

# Crear agente SQL
chain = create_sql_agent(db=db, llm=llm, agent_type=AgentType.ZERO_SHOT_REACT_DESCRIPTION, verbose=True, agent_executor_kwargs={"handle_parsing_errors": True})

def clear_query(query):
    """Extrae solo la consulta SQL del texto generado."""
    prompt = ("Dado el siguiente texto que contiene una consulta SQL.\n"
              f"{query}\n"
              "Extrae solo la consulta SQL sin modificarla.")
    response = llm.invoke(prompt)
    return response.content

def generate_answer(query, question, result):
    """Genera la respuesta en base a la consulta SQL."""
    prompt = ("Dado la siguiente pregunta sobre tesis de geología, su consulta SQL y el resultado SQL, responde en español.\n\n"
              f'Pregunta: {question}\n'
              f'Consulta SQL: {query}\n'
              f'Resultado SQL: {result}')
    response = llm.invoke(prompt)
    return response.content

# Interfaz de usuario
def main():
    st.write("Escribe tu pregunta sobre tesis de geología:")
    user_input = st.chat_input("Haz una pregunta sobre tesis...")

    if user_input:
        with st.spinner("Buscando respuesta..."):
            query = chain.invoke({"input": user_input, "table_info": db.get_table_info()})
            query = clear_query(query)
            result = db.run(query)
            answer = generate_answer(query=query, question=user_input, result=result)

        st.subheader("Respuesta del Chatbot:")
        st.write(answer)

if __name__ == "__main__":
    main()
