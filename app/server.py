# server.py

from model import Model
from flask import Flask, request, jsonify, send_file, make_response

app = Flask(__name__)
model = Model()

# json
def json_response(obj, code = 200):
	response = make_response(jsonify(obj), code)
	response.headers["Content-Type"] = "application/json"
	return response

# model
@app.route('/model', methods=['POST'])
def run_model():
	title = request.form.get('title')
	text = request.form.get('text')
	try:
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
		return send_file('index.html')
	except Exception as e:
		return json_response({
			"status": False,
			"data": {
				"message": str(e)
			}
		}, 500)

if __name__ == '__main__':
	print('FAKE NEWS CLASSIFIER')
	print('initializing model')
	model.initialize('data.csv')
	print('running flask')
	app.run(threaded=True, port=8000)
