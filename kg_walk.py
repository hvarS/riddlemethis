# %%
import requests
import heapq
from tqdm import tqdm

# %%
words = [f.split(',')[0].strip() for f in open('generations/rmt_test.csv','r')][1:]

# %%

def get_related_concepts(word, limit=10):
    base_url = 'http://api.conceptnet.io/'
    search_url = f'{base_url}c/en/{word}?limit={limit}'
    related_concepts = []
    w_concepts = []

    response = requests.get(search_url)
    if response.status_code == 200:
        data = response.json()

        edges = data['edges']
        for edge in edges:
            related_concepts.append(edge['end']['label'])
            w_concepts.append(edge['weight'])
    return related_concepts, w_concepts

def bfs(word, depth=3, limit=10):
    visited = set()
    queue = [(word, 0)]
    concepts = []
    
    while queue:
        current_word, current_depth = queue.pop(0)
        if current_word not in visited:
            visited.add(current_word)
            concepts.append(current_word)

            if current_depth < depth:
                related,_ = get_related_concepts(current_word, limit)
                for related_word in related:
                    queue.append((related_word, current_depth + 1))

    return concepts


def dfs(word, depth=3, limit=10, concepts = [], visited=None,):
    if visited is None:
        visited = set()

    if word not in visited:
        visited.add(word)
        concepts.append(word)

        if depth > 0:
            related,_ = get_related_concepts(word, limit)
            for related_word in related:
                dfs(related_word, depth - 1, limit,concepts,  visited)

    return

def priority(starting_word, limit=10):
    priority_queue = []
    visited = set()
    concepts = []
    heapq.heappush(priority_queue, (0, starting_word))
    ct = 0
    while priority_queue:
        current_w, current_word = heapq.heappop(priority_queue)
        ct += 1
        if ct > limit:
            break

        if current_word not in visited:
            visited.add(current_word)
            concepts.append(current_word)

            related,w = get_related_concepts(current_word, limit)
            for related_word,wt in zip(related,w):
                heapq.heappush(priority_queue, (w, related_word))
    return concepts

# %%
towrite = open('rmt_test_w_concept.csv','w')

# %%
final_test = {
    "word":words
}

# %%
bfs_net = []
dfs_net = []
pq_net = []

# %%
for word in tqdm(words):
    bfs_concepts = bfs(word, depth=3, limit=10)[1:]
    dfs_concepts = []
    dfs(word, 10, 3, dfs_concepts)
    dfs_concepts = dfs_concepts[1:]
    pq_concepts = priority(word, 20)[1:]
    bfs_net.append(bfs_concepts)
    dfs_net.append(dfs_concepts)
    pq_net.append(pq_concepts)

# %%
final_test['BFSConcepts'] = bfs_net
final_test['DFSConcepts'] = dfs_net
final_test['PQConcepts'] = pq_net

# %%
riddles = [f.split(',')[1].strip() for f in open('generations/rmt_test.csv','r')][1:]
classes = [f.split(',')[1].strip() for f in open('generations/rmt_test.csv','r')][1:]
final_test['Riddle'] = riddles
final_test['Class'] = classes

# %%
import pandas as pd
df = pd.DataFrame.from_dict(final_test)
df.to_csv('rmt_test_w_concepts.csv')


