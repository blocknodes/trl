---
tags:
- trl
---

# DeepMath-103K Dataset

## Summary

[DeepMath-103K](https://huggingface.co/datasets/zwhe99/DeepMath-103K) is meticulously curated to push the boundaries of mathematical reasoning in language models.

## Data Structure

- **Format**: [Conversational](https://huggingface.co/docs/trl/main/dataset_formats#conversational)
- **Type**: [Prompt-only](https://huggingface.co/docs/trl/main/dataset_formats#prompt-only)

Column:
- `"prompt"`: The input question.
- `"solution"`: The solution to the math problem.

## Generation script

The script used to generate this dataset can be found [here](https://github.com/huggingface/trl/blob/main/examples/datasets/deepmath_103k.py).
