from flask import Flask, render_template, request
import predict

app = Flask(__name__)

@app.route('/', methods = ['GET'])
def home():
    return render_template("index.html")


@app.route('/', methods = ['POST'])
def TrainModel():    

    age = int(request.form.get("age"))
    sex = int(request.form.get("sex"))
    cp = int(request.form.get("cp"))
    trestbps = int(request.form.get("trestbps"))
    chol = int(request.form.get("chol"))
    fbs = int(request.form.get("fbs"))
    restecg = int(request.form.get("restecg"))
    thalach = int(request.form.get("thalach"))
    exang = int(request.form.get("exang"))
    oldpeak = int(request.form.get("oldpeak"))
    slope = int(request.form.get("slope"))
    ca = int(request.form.get("ca"))
    thal = int(request.form.get("thal"))

    prediction = predict.predictHeartDisease(age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal)
    print(prediction)
    
    return render_template("index.html", x = prediction)


if __name__ == '__main__':
    app.run(port = 3000, debug = True)