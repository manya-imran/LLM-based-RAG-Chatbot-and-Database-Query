from dotenv import load_dotenv
from flask import Flask, request, jsonify
from chatbot import RAG, create_connection, chatbot, query_customer, update_customer

load_dotenv()

app = Flask(__name__)
docs_dir = '/app/Documents'  # Company Documents
conn = create_connection("customers.db")
rag = RAG(
    docs_dir='Documents/',  # Name of the directory where the documents are located
    n_retrievals=1,  # Number of documents returned by the search
    chat_max_tokens=300,  # Maximum number of tokens that can be used in chat memory
    temperature=1.2,  # How creative the response will be
)


@app.route('/get_customer', methods=['GET'])
def get_customer():
    username = request.args.get('username')
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
        return jsonify(response)
    else:
        return jsonify({"error": "Customer not found"}), 404

@app.route('/update_customer', methods=['POST'])
def update_customer():
    data = request.get_json()
    username = data.get('username')
    field = data.get('field')
    new_value = data.get('new_value')
    success = update_customer(conn, username, field, new_value)
    if success:
        return jsonify({"status": "Customer updated"})
    else:
        return jsonify({"error": "Customer not found or update failed"}), 400

@app.route('/ask_question', methods=['GET'])
def ask_question():
    question = request.args.get('question')
    response = rag.ask(question)
    return jsonify({"response": response})

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000)
