import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

# Load the tokenizer and model
model_name = "meta-llama/Meta-Llama-3-8B"
model_path = "./outputs/checkpoint-test/step-1"
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = AutoModelForCausalLM.from_pretrained(model_path)

device = torch.device("cuda")
model.to(device)

# Define the input string
input_text = "Once upon a time in a land far, far away"

# # Tokenize the input
inputs = tokenizer(input_text, return_tensors="pt").to(device)

# # Generate output
outputs = model.generate(
    **inputs,
    max_length=200,
    do_sample=False,
    # return_dict_in_generate=True,
    output_logits=True,
    output_scores=True
)
# print(outputs.logits)

# # Decode the output
decoded_output = tokenizer.decode(outputs[0], skip_special_tokens=True)

# # Print the result
print("Generated text:", decoded_output)

# logits = torch.randn((batch_size, vocab_size)
# log_probs = torch.nn.functional.log_softmax(logits, dim=-1)
