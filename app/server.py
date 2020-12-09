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
def model_predict():
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
		prediction = model.predict(title, text, model_type)
		return succeed_response({
			"title": title,
			"model": model_type,
			"prediction": prediction
		})
	except Exception as e:
		print(e)
		return fail_response(str(e))
@app.route('/model/bert', methods=['GET', 'POST'])
def model_bert():
	if request.method == 'POST':
		try:
			action = request.form.get('action')
			if action == "" or action == None:
				return fail_response("missing parameter: action", 400)
			if action == "init":
				model.bert_init()
				return succeed_response({ "message": "BERT initialized", "bert_status": model.bert_status })
			elif action == "load":
				model.bert_load("bert/model")
				return succeed_response({ "message": "BERT loaded", "bert_status": model.bert_status })
			elif action == "train":
				return succeed_response({ "message": "BERT trained", "bert_status": model.bert_status })
		except Exception as e:
			print(e)
			return fail_response(str(e))
	else:
		try:
			return render_template('bert.html', bert_status=model.bert_status)
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
model.train('data/alldata.csv')
if __name__ == '__main__':
	print('running flask')
	app.run(threaded=True, port=8010)