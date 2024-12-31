#%%
from huggingface_hub import login, hf_hub_download
from datasets import load_dataset
import pandas as pd
from pathlib import Path
from transformers import GPT2Tokenizer

#%%
# input the hf token here
# TOKEN = ""

# login(token=TOKEN, add_to_git_credential=True)

# load dataset by api
# dataset = load_dataset("mteb/tweet_sentiment_extraction")
# df = pd.DataFrame(dataset['train'])

# splits = {'train': 'train.jsonl', 'test': 'test.jsonl'}
# df = pd.read_json("hf://datasets/mteb/tweet_sentiment_extraction/" +
#                   splits["train"])

# load dataset from local
path_data = Path("./data")
assert (path_data)
df_train = pd.read_json(path_data / "train.jsonl", lines=True)
df_test = pd.read_json(path_data / "test.jsonl", lines=True)

#%%
tokenizer = GPT2Tokenizer.from_pretrained("gpt2")
tokenizer.pad_token = tokenizer.eos_token
