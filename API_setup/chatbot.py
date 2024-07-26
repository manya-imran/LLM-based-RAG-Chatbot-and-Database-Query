import os
import openai
import sqlite3
from sqlite3 import Error
from langchain.schema.messages import HumanMessage, SystemMessage
from langchain.prompts import ChatPromptTemplate
from langchain_chroma import Chroma
from langchain.chat_models import ChatOpenAI
from langchain.embeddings import OpenAIEmbeddings
from langchain.document_loaders import DirectoryLoader
from langchain.document_loaders import UnstructuredWordDocumentLoader
from langchain.memory import ConversationTokenBufferMemory
from langchain_core.callbacks import FileCallbackHandler, StdOutCallbackHandler
from langchain_core.output_parsers import StrOutputParser
from loguru import logger
from unstructured.partition.docx import partition_docx
from flask import Flask, request, jsonify

logfile = "output1.log"
logger.add(logfile, colorize=True, enqueue=True)
handler_1 = logger.add(logfile)
handler_2 = logger.add(lambda msg: print(msg, end=""))

os.environ['OPENAI_API_KEY'] = 'OPENAI_API_KEY'
openai.api_key = os.getenv('OPENAI_API_KEY')

class RAG:
    def __init__(self, docs_dir: str, n_retrievals: int = 6, chat_max_tokens: int = 300, model_name="gpt-3.5-turbo", temperature: float = 0.7):
        self.__model = self.__set_llm_model(model_name, temperature)
        self.__docs_list = self.__get_docs_list(docs_dir)
        self.db = self.__set_chroma_db()
        self.__retriever = self.__set_retriever(k=n_retrievals)
        self.__chat_history = self.__set_chat_history(max_token_limit=chat_max_tokens)

    def __set_llm_model(self, model_name="gpt-3.5-turbo", temperature: float = 0.7):
        return ChatOpenAI(model_name=model_name, temperature=temperature)

    def __get_docs_list(self, docs_dir: str) -> list:
        loader = DirectoryLoader(docs_dir, recursive=True, show_progress=True, glob="*.docx", loader_cls=UnstructuredWordDocumentLoader)
        docs_list = loader.load_and_split()
        return docs_list

    def __set_chroma_db(self):
        embeddings = OpenAIEmbeddings()
        db = Chroma.from_documents(self.__docs_list, embeddings)
        return db

    def __set_retriever(self, k: int = 4):
        metadata_field_info = [
            {
                "name": "source",
                "description": "The directory path where the document is located",
                "type": "string"
            },
        ]
        document_content_description = "Personal documents"
        retriever = self.db.as_retriever(search_kwargs={"k": k})
        return retriever

    def __set_chat_history(self, max_token_limit: int = 300):
        return ConversationTokenBufferMemory(llm=self.__model, max_token_limit=max_token_limit, return_messages=True)

    def ask(self, question: str):
        relevant_docs = self.db.similarity_search(question)
        if relevant_docs:
            context = "\n".join([doc.page_content for doc in relevant_docs])
            messages = [
                SystemMessage(content="You are a helpful assistant."),
                HumanMessage(content=f"Context: {context}\n\nQuestion: {question}")
            ]
            response = self.__model(messages)
            response = response.content
            logger.info(f"Accessed: RAG \nResponse: {response}")
        else:
            response = self.__model([HumanMessage(content=question)])
            response = response.content
            logger.info(f"Accessed: Q/A model \nResponse: : {response}")
        return response

def create_connection(db_file):
    conn = None
    try:
        conn = sqlite3.connect(db_file)
        drop_table(conn)
        create_table(conn)
        add_sample_customers(conn)
        return conn
    except Error as e:
        print(e)
    return conn

def drop_table(conn):
    try:
        sql_drop_customers_table = """ DROP TABLE IF EXISTS customers; """
        c = conn.cursor()
        c.execute(sql_drop_customers_table)
    except Error as e:
        print(e)

def create_table(conn):
    try:
        sql_create_customers_table = """ CREATE TABLE IF NOT EXISTS customers (
                                            username text NOT NULL,
                                            password text NOT NULL,
                                            name text NOT NULL,
                                            email text NOT NULL,
                                            address text NOT NULL,
                                            account text NOT NULL,
                                            serviceUse text NOT NULL
                                        ); """
        c = conn.cursor()
        c.execute(sql_create_customers_table)
    except Error as e:
        print(e)

def authenticate_customer(conn, account_number, password):
    try:
        cursor = conn.cursor()
        cursor.execute("SELECT * FROM customers WHERE account = ? AND password = ?", (account_number, password))
        customer = cursor.fetchone()
        if customer:
            return True
        else:
            return False
    except Error as e:
        print("Error:", e)
        return False

def format_response(response):
    formatted_response = ""
    for key, value in response.items():
        formatted_response += f"{key}: {value}\n"
    return formatted_response

def add_sample_customers(conn):
    try:
        sql_insert_customer = """ INSERT INTO customers (username, password, name, email, address, account, serviceUse)
                                  VALUES (?, ?, ?, ?, ?, ?, ?) """
        customers = [
            ('johndoe123','secretkey76','John Doe', 'john@yahoo.com', '123 Elm St', '234567','Enterprise Plan'),
            ('janes_acc2','janes_pass45','Jane Doe', 'jane@gmail.com', '456 Oak St', '739200','Family Standard')
        ]
        c = conn.cursor()
        c.executemany(sql_insert_customer, customers)
        conn.commit()
    except Error as e:
        print(e)

def query_customer(conn, username):
    print("Verification required")
    while True:
        account_number = input("Enter your account number: ")
        password = input("Enter your password: ")
        if authenticate_customer(conn, account_number, password):
            print("Authentication successful!")
            break
        else:
            print("Invalid credentials. Please try again.")
            break
        return None

    try:
        sql_select_customer = f""" SELECT * FROM customers WHERE username = ? """
        c = conn.cursor()
        c.execute(sql_select_customer, (username,))
        return c.fetchone()
    except Error as e:
        print(e)
        return None

def update_customer(conn, username, field, new_value):
    print("Verification required")
    while True:
        account_number = input("Enter your account number: ")
        password = input("Enter your password: ")
        if authenticate_customer(conn, account_number, password):
            print("Authentication successful!")
            break
        else:
            print("Invalid credentials. Please try again.")
            break
        return None
    try:
        sql_update_customer = f""" UPDATE customers
                                   SET {field} = ?
                                   WHERE username = ? """
        c = conn.cursor()
        c.execute(sql_update_customer, (new_value, username))
        conn.commit()
    except Error as e:
        print(e)
        return False
    return True

def chatbot(user_input, conn, rag):
    if user_input.startswith("get customer"):
        username = user_input.split(" ")[-1]
        customer = query_customer(conn, username)
        if customer:
            response = {
                "username": customer[0],
                "password": customer[1],
                "name": customer[2],
                "email": customer[3],
                "address": customer[4],
                "account": customer[5],
                "serviceUse": customer[6]
            }
            logger.info(f"Accessed: DB GET \nResponse: {format_response(response)}" )
            return format_response(response)
        else:
            return {"error": "Customer not found"}
    elif user_input.startswith("update customer"):
        parts = user_input.split(" ")
        username = parts[2]
        field = parts[3]
        print(username, field)
        new_value = " ".join(parts[4:])
        success = update_customer(conn, username, field, new_value)
        if success:
            logger.info("Accessed: DB UPDATE \nResponse: Customer updated")
            return {"status": "Customer updated"}
        else:
            logger.info("Accessed: DB UPDATE \nResponse: Customer not found or update failed")
            return {"error": "Customer not found or update failed"}
    else:
        response = rag.ask(user_input)
        return response
