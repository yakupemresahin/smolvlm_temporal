from mlx_lm import load, generate

model, tokenizer = load("HuggingFaceTB/SmolLM2-1.7B-Instruct-Q8-mlx")
response = generate(model, tokenizer, prompt="hello", verbose=True)

print(response)