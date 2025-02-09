## Chai LLM

Training receipes used for CHAI conversational chatbot. 

## Dataset
Converted from https://huggingface.co/ChaiML
- 50k SFT Data (from https://huggingface.co/datasets/ChaiML/50k_duduk_convo)
- 20k Preference Data (from ChaiML/public_preference_data_20k)

## Training Pipeline
Base Model: Mistral-Nemo-13B

#### SFT + RLHF
```
bash run_sft.sh
bash run_orpo.sh
```
#### Merge with public solution
SLERP weight merging with public model from Chaiverse (rica40325/10_14dpo)

## Compute
4 x A100; ~10 GPU hrs
