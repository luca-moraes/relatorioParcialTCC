from flask import Flask, render_template, request, jsonify
import textwrap
import FlaskDcf as dcf
from FlaskModels import QuestionsOnly, AnswerParams, RefResponse
import FlaskNLP as myNLP

app = Flask(__name__)

questions = dcf.loadQuestionsOnly()
questionsEn = dcf.loadEnQuestionsOnly()
questionsEs = dcf.loadEsQuestionsOnly()

@app.route('/')
def index():
    return render_template('index.html', questions=questions)

@app.route('/english')
def index_en():
    return render_template('indexEn.html', questionsEn=questionsEn)

@app.route('/espanol')
def index_es():
    return render_template('indexEs.html', questionsEs=questionsEs)

@app.route('/submit', methods=['POST'])
def submit():
    question_id = int(request.form['number_question'])
    user_answer = request.form['answer']
    
    question = next((q for q in questions if q.number_question == question_id), None)
    refs = question.reference_responses
    
    answer = myNLP.process_questions(user_answer, refs)
    
    nota_final = answer.nota_final
    
    if nota_final > 2:
        nota_final = myNLP.normalize(nota_final, 0, 3, 0, 10)
        message = f"Correto!<br> Sua nota é: {nota_final:.2f}.<br>Seu parâmetros foram:<br> Frequência de termos -> {answer.frequencia_termos:.2f}<br> Distância de Levenshtein -> {answer.levenshtein_distancia:.2f}<br> Semântica BERT -> {answer.bert_semantica:.2f}"
        color_class = "correct"
    else:
        message = f"Incorreto!<br> Uma resposta correta poderia ser: {refs[0].reference_response}<br> Sua nota seria: {nota_final:.2f}.<br>Seu parâmetros foram:<br> Frequência de termos -> {answer.frequencia_termos:.2f}<br> Distância de Levenshtein -> {answer.levenshtein_distancia:.2f}<br> Semântica BERT -> {answer.bert_semantica:.2f}"
        color_class = "incorrect"
    
    wrapped_message = textwrap.fill(message, width=40)
    
    return jsonify({"message": wrapped_message, "color_class": color_class})

@app.route('/submitEn', methods=['POST'])
def submitEn():
    question_id = int(request.form['number_question'])
    user_answer = request.form['answer']
    
    question = next((q for q in questionsEn if q.number_question == question_id), None)
    refs = question.reference_responses
    
    answer = myNLP.process_questions_en(user_answer, refs)
    
    nota_final = answer.nota_final
    
    if nota_final > 3:
        nota_final = myNLP.normalize(nota_final, 0, 5, 0, 10)
        message = f"Correto!<br> Sua nota é: {nota_final:.2f}.<br>Seu parâmetros foram:<br> Frequência de termos -> {answer.frequencia_termos:.2f}<br> Distância de Levenshtein -> {answer.levenshtein_distancia:.2f}<br> Semântica BERT -> {answer.bert_semantica:.2f}"
        color_class = "correct"
    else:
        message = f"Incorreto!<br> Uma resposta correta poderia ser: {refs[0].reference_response}<br> Sua nota seria: {nota_final:.2f}.<br>Seu parâmetros foram:<br> Frequência de termos -> {answer.frequencia_termos:.2f}<br> Distância de Levenshtein -> {answer.levenshtein_distancia:.2f}<br> Semântica BERT -> {answer.bert_semantica:.2f}"
        color_class = "incorrect"
    
    wrapped_message = textwrap.fill(message, width=40)
    
    return jsonify({"message": wrapped_message, "color_class": color_class})

@app.route('/submitEs', methods=['POST'])
def submitEs():
    question_id = int(request.form['number_question'])
    user_answer = request.form['answer']
    
    question = next((q for q in questionsEs if q.number_question == question_id), None)
    refs = question.reference_responses
    
    answer = myNLP.process_questions_es(user_answer, refs)
    
    nota_final = answer.nota_final
    
    if nota_final > 3:
        nota_final = myNLP.normalize(nota_final, 0, 5, 0, 10)
        message = f"Correto!<br> Sua nota é: {nota_final:.2f}.<br>Seu parâmetros foram:<br> Frequência de termos -> {answer.frequencia_termos:.2f}<br> Distância de Levenshtein -> {answer.levenshtein_distancia:.2f}<br> Semântica BERT -> {answer.bert_semantica:.2f}"
        color_class = "correct"
    else:
        message = f"Incorreto!<br> Uma resposta correta poderia ser: {refs[0].reference_response}<br> Sua nota seria: {nota_final:.2f}.<br>Seu parâmetros foram:<br> Frequência de termos -> {answer.frequencia_termos:.2f}<br> Distância de Levenshtein -> {answer.levenshtein_distancia:.2f}<br> Semântica BERT -> {answer.bert_semantica:.2f}"
        color_class = "incorrect"
    
    wrapped_message = textwrap.fill(message, width=40)
    
    return jsonify({"message": wrapped_message, "color_class": color_class})

if __name__ == '__main__':
    app.run(debug=True)
