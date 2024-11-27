from mlx_lm import load, generate

model, tokenizer = load("mlx-community/SmolLM-1.7B-Instruct-8bit")
response = generate(model, tokenizer, prompt="hello", verbose=True)

print(response)