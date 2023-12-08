from transformers import AutoModelForCausalLM, AutoTokenizer
from transformers import DownloadManager

# Configurable model name - Replace "gpt2" with the name of your desired model in quotes
MODEL_NAME = "gpt2"

# Download model and tokenizer locally
download_manager = DownloadManager(cache_dir="~/.cache/huggingface")
model = AutoModelForCausalLM.from_pretrained(MODEL_NAME, local_files_only=True, download_manager=download_manager)
tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME, local_files_only=True, download_manager=download_manager)

def generate_response(prompt, patient_data=None):
    try:
        formatted_prompt = f"{prompt} {patient_data}" if patient_data else prompt

        inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
        outputs = model.generate(inputs, max_length=150, num_return_sequences=1)

        response = tokenizer.decode(outputs[0])
        return response
    except Exception as e:
        print(f"Error generating response: {e}")
        return "Error: Could not generate response."
