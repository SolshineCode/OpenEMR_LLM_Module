from transformers import AutoModelForCausalLM, AutoTokenizer

# Configurable model name
MODEL_NAME = "gpt2" 

def generate_response(prompt, patient_data=None):
  try:
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForCausalLM.from_pretrained(MODEL_NAME)

    formatted_prompt = f"{prompt} {patient_data}" if patient_data else prompt

    inputs = tokenizer.encode(formatted_prompt, return_tensors="pt")
    outputs = model.generate(inputs, max_length=150, num_return_sequences=1)

    response = tokenizer.decode(outputs[0])
    return response
  except Exception as e:
    print(f"Error generating response: {e}")
    return "Error: Could not generate response."
