# server.py

from model import ModelV2
from flask import Flask, request, jsonify, send_file, make_response, render_template

app = Flask(__name__, static_url_path='', 
            static_folder='static',
            template_folder='templates')
model = ModelV2()

# json
def json_response(obj, code):
	response = make_response(jsonify(obj), code)
	response.headers["Content-Type"] = "application/json"
	return response
def fail_response(msg, code = 500):
	return json_response({
		"status": False,
		"data": {
			"message": msg
		}
	}, code)
def succeed_response(data, code = 200):
	return json_response({
		"status": True,
		"data": data
	}, code)

# model
@app.route('/model', methods=['POST'])
def run_model():
	try:
		title = request.form.get('title')
		text = request.form.get('text')
		model_type = request.form.get('model')
		if text == "" or text == None:
			return fail_response("missing parameter: text", 400)
		if title == "" or title == None:
			return fail_response("missing parameter: title", 400)
		if model_type == "" or model_type == None:
			return fail_response("missing parameter: model_type", 400)
		text = model.preprocess_text(text)
		prediction = model.predict(title, text, model_type)
		return succeed_response({
			"title": title,
			"model": model_type,
			"prediction": prediction
		})
	except Exception as e:
		print(e)
		return fail_response(str(e))

# index
@app.route('/')
def index():
	try:
		return render_template('index.html')
	except Exception as e:
		print(e)
		return fail_response(str(e))

# main
print('FAKE NEWS CLASSIFIER')
print('training model')
model.train('alldata.csv')
if __name__ == '__main__':
	print('running flask')
	app.run(threaded=True, port=8010)