# server.py

from model import Model
from flask import Flask, request, jsonify, send_file, make_response, render_template

app = Flask(__name__, static_url_path='', 
            static_folder='static',
            template_folder='templates')
model = Model()

# json
def json_response(obj, code = 200):
	response = make_response(jsonify(obj), code)
	response.headers["Content-Type"] = "application/json"
	print(response)
	return response

# model
@app.route('/model', methods=['POST'])
def run_model():
	try:
		title = request.form.get('title')
		text = request.form.get('text')
		if text == "":
			return json_response({
				"status": False,
				"data": {
					"message": "missing parameter: text"
				}
			}, 400)
		text = model.preprocess_text(text)
		prediction = model.predict(title, text)
		return json_response({
			"status": True,
			"data": {
				"title": title,
				"prediction": prediction
			}
		})
	except Exception as e:
		return json_response({
			"status": False,
			"data": {
				"message": str(e)
			}
		}, 500)

# index
@app.route('/')
def index():
	try:
		return render_template('index.html')
	except Exception as e:
		return json_response({
			"status": False,
			"data": {
				"message": str(e)
			}
		}, 500)

# main
print('FAKE NEWS CLASSIFIER')
print('training model')
model.train('data.csv')
if __name__ == '__main__':
	print('running flask')
	app.run(threaded=True, port=8000)