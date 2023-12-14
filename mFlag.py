import argparse
from tqdm import tqdm
import pandas as pd
from mFlag.model import MultiFigurativeGeneration
from mFlag.tokenization_mflag import MFlagTokenizerFast



parser = argparse.ArgumentParser(description='Argument Parser for Training TRex Model')
parser.add_argument('--gen_file', type=str, required=True, help='Location of the Generated file: "Model with Use the GenRiddles Column"')
parser.add_argument('--out_file', type=str, required=True, help='Name of the file that will store the output generations')
parser.add_argument('--literal', action='store_true', help='Option for <literal>')
parser.add_argument('--hyperbole', action='store_true', help='Option for <hyperbole>')
parser.add_argument('--idiom', action='store_true', help='Option for <idiom>')
parser.add_argument('--sarcasm', action='store_true', help='Option for <sarcasm>')
parser.add_argument('--metaphor', action='store_true', help='Option for <metaphor>')

# Setting the default option to <literal>
parser.set_defaults(literal=True)

args = parser.parse_args()

df = pd.read_csv(f"{args.gen_file}")
sentences = list(df["GenRiddle"])
# %%
# sentences = [
# "When you close your eyes and take your flight,\nInto the world of the subconscious, I take my might.\nOften filled with wonder, or sometimes with fear,\nWhat am I, the visions that appear so clear?",
# "When hearts are heavy, and grudges grow,I offer healing, a path to let go.I mend relationships, set the spirit free,What am I, this act of mercy?",
# "I'm slender and long, with a lead so fine,I help you write, sketch, and underline.With an eraser for mistakes, I am your tool,What am I, this writing and drawing jewel?",
# "In canyons and hills, I'm often found, I mimic your voice, bouncing all around. I repeat your words, with a distant flow,What am I, this natural audio echo?",
# "You stand before me, a twin so clear,\nReflecting your image, I'm always near.\nThough I have no life, I mimic your grace,\nWhat am I, this reflective surface space?",
# "I'm filled with pages of stories and lore,\nA gateway to knowledge, to worlds to explore.\nYou turn my pages, and get lost in the plot,\nWhat am I, this literary treasure trove?",
# "I rumble and roar, with molten fire,\nMy eruptions are fierce, and my ash goes higher.\nI shape the land with my forceful might,\nWhat am I, this geological sight?",
# "I'm where you think, feel, and create,\nYour thoughts and dreams, I help navigate.\nYou ponder and reason, in me you unwind,\nWhat am I, this enigmatic realm of your mind?",
# "I'm a source of power, stored in a cell,\nIn devices I work, oh, I serve them well.\nFrom toys to your phone, I give energy the right,\nWhat am I, this electrical might?",
# "I'm worn on the head, I change your hair's look,\nFrom long to short, from blonde to brook.\nI'm not real hair, yet I make you feel good,\nWhat am I, this accessory understood?",
# "I'm not a rodent, but I'm part of your tech,\nI click and I scroll, I'm what you select.\nOn a screen I move, with a click or a tap,\nWhat am I, this device in your lap?",
# "I'm made in the winter, with snowballs galore,\nA carrot for a nose, and a hat I wore.\nI stand in your yard, with a smile so wide,\nWhat am I, this frosty guide?",
# "I'm what you read, watch, and hear every day,\nCurrent events and stories, I'm the mainstay.\nI inform, entertain, or sometimes confuse,\nWhat am I, this daily news?",
# "I'm a creature feared, with venomous might, My hourglass marking, a warning in sight. What am I, lurking in shadows so wide, A spinner of webs, with darkness as my guide?",
# "I'm a form of radiation, unseen by the eye, Through objects I pass, revealing what lies inside. Doctors use me to diagnose and assess, What am I, a glimpse into the body's finesse?",
# "I\'m an exact copy, a genetic twin, From another organism, I begin. A controversial topic, debated and known, What am I, a duplicate, not quite on my own?",
# "I'm a creature of the night, silent in flight,\nMy head can spin, and my vision's so bright.\nWith feathers so soft and eyes that appeal,\nWhat am I, this nocturnal bird of steel?",
# "I'm always ahead, just out of reach,\nA day in the future, with lessons to teach.\nYou'll find me on calendars, but not in today,\nWhat am I, this time just a day away?",
# "I'm at the tip of your toe, a part so small,\nI can be trimmed or painted, it's your call.\nNot a finger, but part of your foot's detail,\nWhat am I, this tiny keratin scale?",
# "I'm a concept, endless, with no final stop,\nNumbers go on, and I'm at the top.\nA symbol like a sideways eight,\nWhat am I, this mathematical state?",
# "I tick and I tock, I measure the day,\nWith hands that move, I show the way.\nFrom the wall or the wrist, I keep you in sync,\nWhat am I, this time-telling link?"
# ]

tokenizer = MFlagTokenizerFast.from_pretrained('laihuiyuan/mFLAG')
model = MultiFigurativeGeneration.from_pretrained('laihuiyuan/mFLAG')


sentences = [sentence.replace('\n','') for sentence in sentences]

gens = []

# special tokens: <literal>, <hyperbole>, <idiom>, <sarcasm>, <metaphor>, or <simile>
spcl_token = "<literal>"
if args.hyperbole:
    spcl_token = "<hyperbole>"
elif args.idiom:
    spcl_token = "<idiom>"
elif args.sarcasm:
    spcl_token = "<sarcasm>"
elif args.metaphor:
    spcl_token = "<metaphor>"
elif args.simile:
    spcl_token = "<simile>"


for sentence in tqdm(sentences):
  inp_ids = tokenizer.encode(sentence, return_tensors="pt")
  # the target figurative form (<sarcasm>)
  fig_ids = tokenizer.encode(spcl_token, add_special_tokens=False, return_tensors="pt")
  outs = model.generate(input_ids=inp_ids[:, 1:], fig_ids=fig_ids, forced_bos_token_id=fig_ids.item(), num_beams=5, max_length=256)
  text = tokenizer.decode(outs[0, 2:].tolist(), skip_special_tokens=True, clean_up_tokenization_spaces=False)
  gens.append(text)


df['GenRiddle'] = gens
pd.DataFrame.from_dict(df).to_csv(f"{args.out_file}",index=False)