# Huge Thanks to TheEdoardo93 for their answer in https://github.com/huggingface/transformers/issues/1725
# Which I used as a baseline working version to patch the Conv-AI onto.

import torch
import random

import torch.nn.functional as F

from itertools import chain
from argparse import ArgumentParser
from transformers import GPT2Tokenizer, GPT2LMHeadModel

tokenizer = GPT2Tokenizer.from_pretrained('gpt2')
model = GPT2LMHeadModel.from_pretrained('gpt2')
model.eval()

username = "<Sean>"
botname = "<Lika>"
starttag = "<"

def sample_sequence(generated, temperature, num_samples):

	outputs = model(generated)
	next_token_logits = outputs[0][:, -1, :] / (temperature if temperature > 0 else 1.)
	probs = F.softmax(next_token_logits, dim=-1)

	next_token = torch.multinomial(probs, num_samples=1)
	generated = torch.cat((generated, next_token), dim=1)

	return generated

def gen_to_text(generated, context_tokens):
	out = generated
	out = out[:, len(context_tokens):].tolist()
	for o in out:
		text = tokenizer.decode(o, clean_up_tokenization_spaces=True)
	if starttag in text:
		text = text[:text.index(starttag)]
	return text.strip()

def get_response(length, temperature, persona, chatlog, num_samples = 1):

	# Tokenize persona and chatlog
	input_sequence = persona+chatlog+[botname+" "]
	context_tokens = list(chain(*[tokenizer.encode(seq, add_special_tokens=False) for seq in input_sequence]))

	# Generate Context?
	context = torch.tensor(context_tokens, dtype=torch.long)
	context = context.unsqueeze(0).repeat(num_samples, 1)
	generated = context

	# Generate Response?
	with torch.no_grad():
		for jj in range(5):
			for _ in range(length):
				generated = sample_sequence(generated, temperature, num_samples)

	# Process Response for output
	return gen_to_text(generated, context_tokens)

if __name__ == "__main__":

	# Parse Args
	parser = ArgumentParser()
	parser.add_argument("--seed", type=int, default=0, help="Seed")
	args = parser.parse_args()

	# Process Random Seed Arg
	if args.seed != 0:
		random.seed(args.seed)
		torch.random.manual_seed(args.seed)
		torch.cuda.manual_seed(args.seed)

	# Varibles used to control generation
	length = 20
	temperature = 0.7
	max_history = 2
	persona = ["I like cake becuase it is sweet.", "I don't like getting caught in the rain.", "I'm 32.", "I am a single woman.", "My name is Lika."]
	persona = [botname+" "+p for p in persona]
	chatlog = []

	# Run Chat Loop
	while True:
		# Get User Input
		sentence = input(username+" ")
		if sentence == "/q":
			break
		
		# Get Bot Response
		chatlog.append(username+" "+sentence)
		response = get_response(length, temperature, persona, chatlog)
		chatlog.append(botname+" "+response)
		chatlog = chatlog[-(2*max_history+1):]
		print(botname+" "+response)
