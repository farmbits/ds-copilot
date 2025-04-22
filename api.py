from flask import Flask, request, Response, jsonify
import json
from aux import *


app = Flask(__name__)

@app.route("/")
def main():
    return "<h1>Copiloto Belagrícola</h1>"


@app.route('/make_question', methods=['POST'])
def make_a_question():
    data = request.get_json()
    question_str = data.get("question_str")
    response = query(question_str)
    return jsonify(response)

if __name__ == "__main__":
    app.run(port=8501, debug=False, host="0.0.0.0")
