# distilabel pipelines

We used [`distilabel`](https://github.com/argilla-io/distilabel) to create several pipelines for generating instruction-following and multi-turn datasets for the post-training of SmolLM2.

> [!NOTE]
> This section is still in WIP. We will upload the rest of the pipelines soon. Thanks for your patience!

# finetuning

We finetune SmolLM2 models on [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) using the alignement handbook, you can find the instructions here: https://github.com/huggingface/alignment-handbook/tree/main/recipes/smollm2#instructions-to-train-smollm2-17b-instruct 
