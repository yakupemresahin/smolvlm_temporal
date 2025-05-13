# Smol VLM Temporal Analysis 🤏

This repository is focused on analyzing the temporal understanding capabilities of SmolVLM through short video sequences. We evaluate how well this compact vision-language model can comprehend and reason about time-based events in brief video clips (max 3 seconds). This repo is forked from [huggingface/smollm](https://github.com/huggingface/smollm).


## Project Overview
We probe SmolVLM, the Vision Language Model of SmolLM2 ecosystem, to understand its temporal reasoning abilities using two categories of actions:

**Reversible actions:** Events that can naturally occur in either direction (e.g., opening/closing a door, turning a light on/off)

**Irreversible actions:** Events with a natural progression that would appear unnatural when reversed (e.g., an apple falling from a tree, a wine glass shattering)


This analysis helps us understand how well SmolVLM perceives temporal sequences. Whether it can distinguish between natural and unnatural event progressions and the limitations of compact VLMs in processing temporal information

## Repository Structure
```
smollm/
├── text/              # SmolLM2 related code and resources
├── vision/            # SmolVLM related code and resources
└── tools/             # Shared utilities and inference 
    ├── smol_tools/    # Lightweight AI-powered tools
    ├── smollm_local_inference/
    ├── smolvlm_local_inference/   # ← Our temporal analysis code is here
        ├── video_samples/         # Directory for test videos
        └── temporal_analysis.py/  # Main script for temporal understanding tests
```

## Getting Started

1. Install the required dependencies:<br>
    ```pip install -r tools/smolvlm_local_inference/requirements.txt```

2. Prepare your test videos:
    - Place your video clipsunder ```tools/smolvlm_local_inference/video_samples/```
    - Videos should be in ```.mp4``` format

3. Run the analysis script:<br>
    ```cd tools/smolvlm_local_inference``` <br>```python temporal_analysis.py```

4. The outputs are saved in ```results.json```

## Resources

### Source
- [Original smollm Repo](https://github.com/huggingface/smollm)

### Documentation
- [SmolLM2 Documentation](text/README.md)
- [SmolVLM Documentation](vision/README.md)
- [Local Inference Guide](tools/README.md)

### Pretrained Models
- [SmolLM2 Models Collection](https://huggingface.co/collections/HuggingFaceTB/smollm2-6723884218bcda64b34d7db9)
- [SmolVLM Model](https://huggingface.co/HuggingFaceTB/SmolVLM-Instruct)

### Datasets
- [SmolTalk](https://huggingface.co/datasets/HuggingFaceTB/smoltalk) - Our instruction-tuning dataset
- [FineMath](https://huggingface.co/datasets/HuggingFaceTB/finemath) - Mathematics pretraining dataset
- [FineWeb-Edu](https://huggingface.co/datasets/HuggingFaceFW/fineweb-edu) - Educational content pretraining dataset