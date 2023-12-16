# RiddleMeThis



1. **riddler.py**: End-to-End file to generate a riddle from a given word/phrase. Usage: 
 ```
 python riddler.py --word <Word/Phrase>
 ```
  - Before running the above, it is important to set the OPENAI_API_KEY environment variable on your system
  
2. **baselines**:
    - `flanT5.py`: Executes the flanT5 baseline on the dataset. Usage: 
      ```
      python flanT5.py --device <GPU_ordinal> --train_loc <TrainingFile Location> --test_loc <TestFile Location> --out_file <Output File to Generate>
      ```
    - `gpt35.py`: Runs the OpenAI GPT3.5 baseline on the dataset. Usage: 
      ```
      python gpt35.py --test_loc <TestFile Location> --out_file <Output File to Generate>
      ```
    - `trex.py`: Executes the TRex baseline on the dataset. Usage: 
      ```
      python trex.py --test_loc <TestFile Location> --out_file <Output File to Generate> --num_pairs <Num of Concept Pairs to Generate>
      ```

3. **data**: Contains the RiddleMeThis dataset divided into `rmt_train.csv`, `rmt_test.csv`, and `rmt_test_concepts.csv` which includes associated priority queue concepts.

4. **data_srcs**: Holds the data sources utilized to create the RiddleMeThis Dataset.

5. **generations**: Houses the generated riddle files.

6. **mFLAG**: Includes the source files to execute the embellishment module over the generated riddles.

7. **notebooks**: Contains notebooks utilized for utility tasks such as dataset unification, concept-based content extraction, dataset statistics, etc.

8. **plots**: Comprises plots utilized in the final submission paper.

9. **auto_evaluate.py**: Utilized for automatic evaluation metrics in the paper. Usage: 



10. **kg_walk.py**: Contains methods for walking over the ConceptNet Graph using BFS, DFS, and Priority Queue methods.

11. **mFlag.py**: Utilized for embellishing the generated base riddles from the open_api model. Usage: 
 ```
 python mFlag.py --gen_file <Path to Generated Riddle File> --out_file <Output File to Generate> --fos
 ```
 `fos` can be from this list: `["<hyperbole>", "<idiom>", "<sarcasm>", "<metaphor>", "<simile>"]`

