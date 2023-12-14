import pandas as pd
import pickle
from tqdm import tqdm
import time
from openai import OpenAI
import argparse

parser = argparse.ArgumentParser(description='Argument Parser for Generating Using the Riddler Model')
parser.add_argument('--test_loc', type=str, required=True, help='Location of the test file')
parser.add_argument('--out_file', type=str, required=True, help='Name of the file that will store the output generations')
args = parser.parse_args()

client = OpenAI()



r = pd.read_csv(f"{args.test_loc}")
words = r['Word']
bfs_concepts = r['BFSConcepts']
dfs_concepts = r['DFSConcepts']
pq_concepts = r['PQConcepts']


openai_generations = []

prompts = []
for i in tqdm(range(len(words))):
  word = words[i]
  cp = pq_concepts[i]
  msg = f"Can you create a riddle for the word: {word} which touches upon these concepts: {eval(cp)[:4]} ?"
  prompts.append(msg)

# %%
for i in tqdm(range(39)):
      ip_prompts = prompts[20*i:20*(i+1)]
      completion = client.completions.create(
            model="gpt-3.5-turbo-instruct",
            prompt=ip_prompts,
            max_tokens=128
      )
      for resp in completion.choices:
            gen = resp.text
            openai_generations.append(gen)
      time.sleep(20)
      

openai_gen = {
    "Words":words,
    "PQConcepts":pq_concepts,
    "Generation":openai_generations
}

pd.DataFrame.from_dict(openai_gen).to_csv('gpt-turbo-gen-pq.csv',index=False)