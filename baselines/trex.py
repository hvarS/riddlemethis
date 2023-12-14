
import pandas as pd
import argparse
from lxml import etree
import requests
import random
from tqdm import tqdm


parser = argparse.ArgumentParser(description='Argument Parser for Training TRex Model')
parser.add_argument('--test_loc', type=str, required=True, help='Location of the test file')
parser.add_argument('--out_file', type=str, required=True, help='Name of the file that will store the output generations')
parser.add_argument('--num_pairs', type=int, default=2, help='Number of Pairs to Generate for the riddle')


args = parser.parse_args()


class Result(object):
    "General result class."
    
    def data_from_element(self, element):
        "Get weight and text from an element"
        text = element.text.strip()
        weight = int(element.attrib['weight'])
        return text, weight
    
    def dict_from_elements(self, elements):
        "Turn a list of elements into a dictionary. K: text, V: weight."
        return dict(self.data_from_element(element) for element in elements)
    
    def dict_from_xml(self, singular, plural):
        "Generate dictionary from result XML. Useful for singular/plural elements."
        results_dict = dict()
        plural_root = self.root.find(plural)
        element_generator = plural_root.iterfind(singular)
        return self.dict_from_elements(element_generator)


# class SingleResult(Result):
#     "Result object for a single word."
    
#     base_url = "http://ngrams.ucd.ie/therex3/common-nouns/member.action?modi={term}&kw={term}&needDisamb=false&xml=true"
    
    

#     def __init__(self, word):
#         query = SingleResult.base_url.format(term=word)
#         response = requests.get(query)
#         self.root = etree.fromstring(response.content)
#         self.categories = self._sort(self.dict_from_xml("Category", "Categories"))
#         self.modifiers = self._sort(self.dict_from_xml("Modifier", "Modifiers"))
#         self.category_heads = self._sort(self.dict_from_xml("CategoryHead", "CategoryHeads"))


#     def _sort(self, x):
#         x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}
#         return x


class mcPair(Result):
    "as attribute as concept"

    starter_url = "http://ngrams.ucd.ie/therex3/common-nouns/member.action?member={sTerm}&kw={sTerm}&needDisamb=false&xml=true"
    new_category_url = "http://ngrams.ucd.ie/therex3/common-nouns/modifier.action?modi={modi}&ref={sTerm}&xml=true"
    new_object_url = "http://ngrams.ucd.ie/therex3/common-nouns/category.action?cate={modi}%3A{concept}&xml=true"

    def _sort(self, x):
        x = {k: v for k, v in sorted(x.items(), key=lambda item: item[1], reverse=True)}
        return x
        
    def __init__(self, searchTerm, numPairs):
        query = mcPair.starter_url.format(sTerm =searchTerm)
        response = requests.get(query)
        self.root = etree.fromstring(response.content)
        self.pairs = []
        for pe in range(numPairs):
            modifiers = self._sort(self.dict_from_xml("Modifier", "Modifiers"))
            modi, _ = random.choice(list(modifiers.items())[:20])
            query = mcPair.new_category_url.format(modi = modi, sTerm =searchTerm)
            response = requests.get(query)
            self.root = etree.fromstring(response.content)
            cHeads = self._sort(self.dict_from_xml("CategoryHead", "CategoryHeads"))
            cHead, _ = random.choice(list(cHeads.items())[:10])
            query = mcPair.new_object_url.format(modi = modi, concept = cHead)
            response = requests.get(query)
            self.root = etree.fromstring(response.content)
            objects = self._sort(self.dict_from_xml("Member", "Members"))
            object, _ = random.choice(list(objects.items())[:5])

            self.pairs.append((modi,object))



starters = list(pd.read_csv(f'{args.test_loc}')['Word'])
gold_riddles = list(pd.read_csv(f'{args.test_loc}')['Riddle'])
wa_riddles = []
wa_answers = []


for starter in tqdm(starters):
    searchTerm = str(starter).lower()
    
    try:
        ep = mcPair(searchTerm,args.num_pairs)
        pairs = ep.pairs
        riddle = "What is "
        for c,p in pairs:
            riddle += f"as {c} as {p}" + " and "
        riddle = riddle[:-4] + " ?"  
    except:
        riddle = ""
    wa_riddles.append(riddle)
    wa_answers.append(searchTerm)


gen = {
    "Word":starters,
    "GoldRiddle": gold_riddles,
    "GenRiddle": wa_riddles
}

pd.DataFrame.from_dict(gen).to_csv(f"{args.out_file}",index=False)