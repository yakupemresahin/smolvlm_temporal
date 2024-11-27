# Local inference

You can use SmolLM2 models locally with frameworks like Transformers.js, llama.cpp, MLX and MLC.

Here you can find the code for running SmolLM locally using each of these libraries. You can also find the conversions of SmolLM & SmolLM2 in these collections: [SmolLM1](https://huggingface.co/collections/HuggingFaceTB/local-smollms-66c0f3b2a15b4eed7fb198d0) and [SmolLM2](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9).

Please first install each library by following its documentation:
- [Transformers.js](https://github.com/huggingface/transformers.js)
- [llama-cpp-python](https://github.com/abetlen/llama-cpp-python)
- [MLX](https://github.com/ml-explore/mlx)
- [MLC](https://github.com/mlc-ai/web-llm)


## Demos
Below are some demos we built for running SmolLM models in the browser:
- [WebGPU demo](https://huggingface.co/spaces/HuggingFaceTB/SmolLM2-1.7B-Instruct-WebGPU ) of SmolLM2 1.7B Instruct powered by Transformers.js and ONNX Runtime Web:  
- [Bunny B1](https://github.com/dottxt-ai/demos/tree/main/its-a-smol-world) mapping natural language requests to local aplication calls using function calling and structured generation by [outlines](https://github.com/dottxt-ai/outlines).
- [Instant SmolLM](https://huggingface.co/spaces/HuggingFaceTB/instant-smollm) powered by MLC for real-time generations of SmolLM-360M-Instruct.

The models are also available on [Ollama](https://ollama.com/library/smollm2) and [PocketPal-AI](https://github.com/a-ghorbani/pocketpal-ai).
