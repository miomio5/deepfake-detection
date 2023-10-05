from flask import Flask , request, jsonify
from model import generate_output

app = Flask(__name__)

@app.route('/classify' , methods =["POST"])
def classify_image():
    if not request.form['image_data']:
        return jsonify({"error":"Image data not found"}),404
    image_data = request.form['image_data']
    response = jsonify(generate_output(image_data))
    
    return response


if __name__ == "__main__":
    app.run()
