from flask import Flask, request, render_template
import run
import json

app = Flask(__name__)   

@app.route("/")
def form():
    return render_template("form.html")

@app.route("/", methods =['POST'])
def guess_tags():
    text = request.form['comments']
    if len(text) > 0:
        model = run.TextPredictionModel.from_artefacts("./saved_models/2022-12-14-14-30-01")
        pred = model.predict([text])
        return str(pred)

@app.route("/predict", methods = ['POST'])
def predict():
    body=json.loads(request.get_data())
    text_list = body['textsToPredict']
    top_k = body['top_k']
    model = run.TextPredictionModel.from_artefacts("./saved_models/2022-12-14-14-30-01")
    pred = model.predict(text_list,top_k=top_k)
    return str(pred)

if __name__ == "__main__":
    app.run(debug=True)
