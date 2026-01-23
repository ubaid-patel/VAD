from flask import Flask, send_from_directory

app = Flask(__name__)

@app.route("/")
def index():
    return send_from_directory(".", "indexa.html")

@app.route("/app.js")
def js():
    return send_from_directory(".", "app.js")

if __name__ == "__main__":
    app.run(host="0.0.0.0", port=5000)
