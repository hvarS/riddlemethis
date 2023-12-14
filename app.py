import streamlit as st
import random

# Function to create a riddle based on input
def create_riddle(word, concepts, selected_embellishments):
    embellishments = {
        'literal': 'What am I?',
        'hyperbole': 'I am larger than life, what could I be?',
        'idiom': 'I am a phrase in disguise, who am I?',
        'simile': 'I am like something else, guess what?',
        'sarcasm': "Oh, I'm definitely not this, right?",
        'metaphor': 'I am something else entirely, what is it?'
    }

    chosen_embellishments = [embellishments[emb] for emb in selected_embellishments]

    riddle = f"I am a {random.choice(chosen_embellishments)}. "
    riddle += f"My word/phrase is {word}. "
    riddle += f"I relate to the following concepts: {', '.join(concepts)}."

    return riddle

# Streamlit app
st.title("Riddle Creator")

word_or_phrase = st.text_input("Enter a word or phrase:")
concepts = st.text_area("Enter concepts separated by commas (e.g., concept1, concept2):")

embellishments = ['literal', 'hyperbole', 'idiom', 'simile', 'sarcasm', 'metaphor']
selected_embellishments = st.multiselect("Select types of embellishments:", embellishments)

if st.button("Create Riddle"):
    if word_or_phrase and concepts and selected_embellishments:
        riddle = create_riddle(word_or_phrase, [c.strip() for c in concepts.split(',')], selected_embellishments)
        st.success("Here's your riddle:")
        st.write(riddle)
    else:
        st.warning("Please fill in all the required fields.")
