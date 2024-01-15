# What needs to be exoerimented with in each experiment : 
# 1. Signature      2. Module parameters        3. Zero-shot or few shot alternative. 

import dspy 
from dspy import dsp
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, BootstrapFewShotWithOptuna
from ethics import data

class CoT(dspy.Module):
    def __init__(self):
        super().__init__()

        self.gen_ans = dspy.ChainOfThought('question -> answer')  # depending on our use case, we can have our signature.
    
    def forward(self, question):
        return self.gen_ans(question=question)

#TODO: Add api key here.
lm = dsp.Cohere(model="command", api_key = "") 

# Change this to add the actual data.
trainset, devset = data.train, data.dev 

dspy.settings.configure(lm=lm)
metric = dspy.evaluate_answer_exact_match

evaluate = Evaluate()

RUN_FROM_SCRATCH = False  # Make it true for zero shot learning. 
NUM_THREADS = 4

if RUN_FROM_SCRATCH:
    config = dict(max_bootstrapped_demos=8, max_labeled_demos=8, num_candidate_programs=10, num_threads=NUM_THREADS)
    teleprompter = BootstrapFewShotWithRandomSearch(metric=metric, **config)
    cot_bs = teleprompter.compile(CoT(), trainset=trainset, valset=devset)
    # cot_bs.save('examples.json')

else:
    cot_bs = CoT()
    cot_bs.load('examples.json')  ## we can add path to our data thing here.

evaluate(cot_bs, devset=devset[:])

lm.inspect_history(n=1)
