from flask import Flask, render_template, request

app = Flask(__name__)

def mi_funcion(numero):
    return numero ** 2

@app.route("/")
def home():
    return render_template("index.html")

@app.route("/calcular", methods=["POST"])
def calcular():
    numero = int(request.form["numero"])
    resultado = mi_funcion(numero)
    return render_template("index.html", resultado=resultado)

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
