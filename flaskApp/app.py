from flask import Flask, render_template, request, jsonify
import textwrap

app = Flask(__name__)

# Lista de questões de exemplo
questions = [
    {"id": 1, "text": "Qual é a capital da França?", "answer": "Paris"},
    {"id": 2, "text": "Qual é a fórmula da água?", "answer": "H2O"},
    {"id": 3, "text": "Quem escreveu 'Dom Quixote'?", "answer": "Miguel de Cervantes"},
]

@app.route('/')
def index():
    return render_template('index.html', questions=questions)

@app.route('/submit', methods=['POST'])
def submit():
    question_id = int(request.form['question_id'])
    user_answer = request.form['answer'].strip().lower()
    
    # Encontra a resposta correta
    question = next((q for q in questions if q["id"] == question_id), None)
    answer = question['answer']
    correct_answer = question['answer'].strip().lower() if question else ""
    
    # Verifica se a resposta está correta
    is_correct = user_answer == correct_answer
    
    # Formata a mensagem com a cor correspondente
    if is_correct:
        message = f"Correto!<br> Pontuação: 10"
        color_class = "correct"
    else:
        message = f"Incorreto!<br> A resposta correta é: {answer}"
        color_class = "incorrect"
    
    # Quebra o texto em linhas para evitar que ele se sobreponha
    wrapped_message = textwrap.fill(message, width=40)
    
    return jsonify({"message": wrapped_message, "color_class": color_class})

if __name__ == '__main__':
    app.run(debug=True)
