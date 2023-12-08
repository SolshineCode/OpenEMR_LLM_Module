from flask import Flask, request, jsonify
from transformers import AutoModelForCausalLM, AutoTokenizer, DownloadManager

app = Flask(__name__)

MODEL_NAME = "gpt2"
download_manager = DownloadManager(cache_dir="~/.cache/huggingface")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

@app.route('/generate', methods=['POST'])
def generate():
    data = request.get_json()
    prompt = data['prompt']
    patient_data = data.get('patient_data')

    formatted_prompt = f"{prompt} {patient_data}" if patient_data else prompt
    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)
    response = tokenizer.decode(outputs[0])

    return jsonify({'response': response})

if __name__ == '__main__':
    app.run(port=5000)
