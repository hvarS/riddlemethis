import nltk
import evaluate
import numpy as np
import torch
import pandas as pd
import argparse
from datasets import load_dataset
from transformers import T5Tokenizer, DataCollatorForSeq2Seq
from transformers import T5ForConditionalGeneration, Seq2SeqTrainingArguments, Seq2SeqTrainer



parser = argparse.ArgumentParser(description='Argument Parser for Training FlanT5')

parser.add_argument('--device', type=int, default=0, help='GPU device number to run on')
parser.add_argument('--train_loc', type=str, required=True, help='Location of the train file')
parser.add_argument('--test_loc', type=str, required=True, help='Location of the test file')
parser.add_argument('--out_file', type=str, required=True, help='Name of the file that will store the output generations')


args = parser.parse_args()



# %%
# Load the tokenizer, model, and data collator
device = f"cuda:{args.device}" if torch.cuda.is_available() else "cpu"
MODEL_NAME = "google/flan-t5-small"

tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME).to(device)
data_collator = DataCollatorForSeq2Seq(tokenizer=tokenizer, model=model)


# Acquire the training data from Hugging Face
gold_riddles = list(pd.read_csv(f"{args.test_loc}")["Riddle"])
rmt = load_dataset("csv", data_files={"train": f"{args.train_loc}", "test": f"{args.test_loc}"})

# %%
# We prefix our tasks with "answer the question"
prefix = "Please generate a riddle for the following word: "

# Define the preprocessing function
def preprocess_function(examples):
   """Add prefix to the sentences, tokenize the text, and set the labels"""
   # The "inputs" are the tokenized answer:
   # print(100*"!")
   # print(type(examples))
   # print(100*"!")
   inputs = [prefix + doc for doc in examples["Word"]]
   model_inputs = tokenizer(inputs, max_length=128, truncation=True)

   # The "labels" are the tokenized outputs:
   labels = tokenizer(text_target=examples["Riddle"],
                      max_length=512,
                      truncation=True)

   model_inputs["labels"] = labels["input_ids"]
   return model_inputs

# %%
# Map the preprocessing function across our dataset
tokenized_dataset = rmt.map(preprocess_function, batched=True)

# %%
nltk.download("punkt", quiet=True)
metric = evaluate.load("rouge")

# %%
def compute_metrics(eval_preds):
   preds, labels = eval_preds

   # decode preds and labels
   labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
   decoded_preds = tokenizer.batch_decode(preds, skip_special_tokens=True)
   decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)

   # rougeLSum expects newline after each sentence
   decoded_preds = ["\n".join(nltk.sent_tokenize(pred.strip())) for pred in decoded_preds]
   decoded_labels = ["\n".join(nltk.sent_tokenize(label.strip())) for label in decoded_labels]

   result = metric.compute(predictions=decoded_preds, references=decoded_labels, use_stemmer=True)

   return result

# %%
# Global Parameters
L_RATE = 3e-4
BATCH_SIZE = 1
PER_DEVICE_EVAL_BATCH = 1
WEIGHT_DECAY = 0.01
SAVE_TOTAL_LIM = 3
NUM_EPOCHS = 1

# Set up training arguments
training_args = Seq2SeqTrainingArguments(
   output_dir="./results",
   evaluation_strategy="epoch",
   learning_rate=L_RATE,
   per_device_train_batch_size=BATCH_SIZE,
   per_device_eval_batch_size=PER_DEVICE_EVAL_BATCH,
   weight_decay=WEIGHT_DECAY,
   save_total_limit=SAVE_TOTAL_LIM,
   num_train_epochs=NUM_EPOCHS,
   predict_with_generate=True,
   push_to_hub=False
)

# %%
trainer = Seq2SeqTrainer(
   model=model,
   args=training_args,
   train_dataset=tokenized_dataset["train"],
   eval_dataset=tokenized_dataset["test"],
   tokenizer=tokenizer,
   data_collator=data_collator,
   compute_metrics=compute_metrics
)

# %%
trainer.train()

# %% [markdown]
# ## Evaluation

# %%

# Generate text on the test dataset.
from transformers import Seq2SeqTrainingArguments, Seq2SeqTrainer
from tqdm import tqdm
test_dataset = tokenized_dataset["test"]

# Define a function to generate text for each example in the test dataset.
def generate_text(dataset):
    generated_outputs = {'input_text':[], 'generated_text':[]}
    for test_sample in tqdm(dataset):
      input_ids = torch.tensor([test_sample["input_ids"]], device = device)
      input_text = test_sample["starter"]
      generated_text = model.generate(input_ids, max_length=128, num_return_sequences=1, no_repeat_ngram_size=2, top_p=0.92)
      generated_text = tokenizer.decode(generated_text[0], skip_special_tokens=True, clean_up_tokenization_spaces=True)
      generated_outputs['input_text'].append(input_text)
      generated_outputs['generated_text'].append(generated_text)

    return generated_outputs

generated_outputs = generate_text(test_dataset)



gen = {
    "Word":generated_outputs['input_text'],
    "GoldRiddle": gold_riddles,
    "GenRiddle": generated_outputs['generated_text'],
}

pd.DataFrame.from_dict(gen).to_csv(f"{args.out_file}",index=False)


# # %%
# with open(f'{args.out_file}','w') as f:
#   for i in range(len(generated_outputs['generated_text'])):
#     starter = generated_outputs['input_text'][i]
#     gold = test_dataset[i]['riddle']
#     pred = generated_outputs['generated_text'][i]
#     line = u'\t\t'.join((starter, pred, gold)).encode('utf-8').strip()
#     f.write(str(line)+'\n')

