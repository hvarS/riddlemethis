
import pandas as pd
import evaluate
import argparse
import math


parser = argparse.ArgumentParser(description='Argument Parser for Evaluating the Generated File Using models')

parser.add_argument('--gen_file', type=str, required=True, help='Location of the Generated file')
args = parser.parse_args()


ev = pd.read_csv(f'{args.gen_file}')


starters, gens, refs = list(ev['Word']), list(ev['GenRiddle']), list(ev['GoldRiddle'])
for i in range(len(gens)): 
    if isinstance(gens[i],str):
        continue
    if math.isnan(gens[i]):
        gens[i] = " "

i = 0
refs = [[ref] for ref in refs]

from nltk.translate.bleu_score import corpus_bleu
from nltk.translate.meteor_score import meteor_score
from rouge_score import rouge_scorer


bleu = evaluate.load("bleu")

# Compute BLEU score
bleu_1 = bleu.compute(predictions=gens, references=refs,max_order = 1)
bleu_2 = bleu.compute(predictions=gens, references=refs,max_order = 2)
bleu_3 = bleu.compute(predictions=gens, references=refs,max_order = 3)
bleu_4 = bleu.compute(predictions=gens, references=refs,max_order = 4)

# # Compute METEOR score
meteor = evaluate.load('meteor')
meteor_score = meteor.compute(predictions=gens, references=refs)

# # Compute Rouge scores

refs_simplified = [ref[0] for ref in refs]

rouge_scorer_instance = rouge_scorer.RougeScorer(['rouge1', 'rouge2', 'rougeL'], use_stemmer=True)
rouge_scores = [rouge_scorer_instance.score(ref, gen) for ref, gen in zip(refs_simplified, gens)]

rouge1_scores = [score['rouge1'].fmeasure for score in rouge_scores]
rouge2_scores = [score['rouge2'].fmeasure for score in rouge_scores]
rougeL_scores = [score['rougeL'].fmeasure for score in rouge_scores]

average_rouge1_score = sum(rouge1_scores) / len(rouge1_scores)
average_rouge2_score = sum(rouge2_scores) / len(rouge2_scores)
average_rougeL_score = sum(rougeL_scores) / len(rougeL_scores)

print("BLEU Scores:")
print("BLEU 1:", bleu_1)
print("BLEU 2:", bleu_2)
print("BLEU 3:", bleu_3)
print("BLEU 4:", bleu_4)
print("METEOR Score:", meteor_score)
print("Rouge-1 Score:", average_rouge1_score)
print("Rouge-2 Score:", average_rouge2_score)
print("Rouge-L Score:", average_rougeL_score)

bertscore = evaluate.load("bertscore")

bertscores = bertscore.compute(predictions=gens, references=refs, lang="en")

# # Print or store the evaluation metrics.
print("BERTScores:")
print(bertscores)
