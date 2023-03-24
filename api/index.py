import requests
import pickle as pkl
from os.path import abspath,join,dirname
from flask import Flask, request, jsonify
dir = dirname(abspath(__file__))

app = Flask(__name__)



# def Loadmodel():
#     with open(join(dir, '..', 'data', 'pidgin_model.pkl'), 'rb') as file:
#         model1 = pkl.load(file)
#     with open(join(dir, '..', 'data', 'pidgin_tokenizer.pkl'), 'rb') as file:
#         token1 = pkl.load(file)
#     with open(join(dir, '..', 'data', 't5_model.pkl'), 'rb') as file:
#         model2 = pkl.load(file)
#     with open(join(dir, '..', 'data', 't5_tokenizer.pkl'), 'rb') as file:
#         token2 = pkl.load(file)
    
#     return model1, token1, model2, token2

# pidgin_model, pidgin_tokenizer, t5_model, t5_tokenizer = Loadmodel()


def predictInput(input:str,model,tokenizer) -> str:
    encodedText = tokenizer.encode(input, return_tensors='pt')
    output = model.generate(encodedText)
    decodedOutput = tokenizer.decode(output[0], skip_special_tokens=True)
    return decodedOutput

def predictOther(input: str, model, tokenizer, Language: str) -> str:
    input = f"translate English to {Language}: {input}"
    encodedText = tokenizer.encode(input, return_tensors='pt')
    output = model.generate(encodedText)
    decodedOutput = tokenizer.decode(output[0], skip_special_tokens=True)
    return decodedOutput


@app.route("/")
def home():
    return f"Hello, World: {dir}"

@app.route('/about')
def about():
    return 'About'

@app.route('/test')
def test():
    payloads = { "inputs" : "how is your family doing?" }
    return payloads

@app.route('/test2')
def test2():
    return "<h1>Everything is good</h1>"    


@app.route('/other')
def other():
    payloads = { "inputs" : "how is your family doing?", "lang" : "German" }
    output = requests.post("https://model-rest-api-route.vercel.app/predict_other",json=payloads)
    return output.json()

@app.route('/predict', methods=["POST"])
def predict():
    # Get the input data from the request body
    data = request.json
    inputText = data["inputs"]
    decodedOutput = predictInput(inputText,pidgin_model,pidgin_tokenizer)
    return jsonify({'prediction': decodedOutput})

@app.route('/predict_other', methods=["POST"])
def predictOtherLangugages():
    # Get the input data from the request body
    data = request.json
    inputText = data["inputs"]
    language = data["lang"]
    decodedOutput = predictOther(inputText,t5_model,t5_tokenizer,language)
    return jsonify({'prediction': decodedOutput})


# if __name__ == "__main__":
#     app.run()

# @app.route('/')
# def home():
#     payloads = { "inputs" : "how is your family doing?" }
#     output = requests.post("https://model-rest-api-route.vercel.app/predict",json=payloads)
#     return output.json()

# def saveFiles():
#     with open(join(dir, '..', 'data', 'pidgin_model.pkl'), 'wb') as file:
#         pkl.dump(pidgin_model,file)
#     with open(join(dir, '..', 'data', 'pidgin_tokenizer.pkl'), 'wb') as file:
#         pkl.dump(pidgin_tokenizer,file)
#     with open(join(dir, '..', 'data', 't5_model.pkl'), 'wb') as file:
#         pkl.dump(t5_model,file)
#     with open(join(dir, '..', 'data', 't5_tokenizer.pkl'), 'wb') as file:
#         pkl.dump(t5_tokenizer,file)


