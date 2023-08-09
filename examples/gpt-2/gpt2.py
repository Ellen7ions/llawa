from transformers import GPT2Tokenizer, GPT2Model
from transformers import AutoTokenizer, AutoModelForCausalLM
from transformers import pipeline
from transformers import GenerationConfig

tokenizer = AutoTokenizer.from_pretrained('gpt2', cache_dir='../pretrained')
model = AutoModelForCausalLM.from_pretrained('gpt2', cache_dir='../pretrained')

cfg = GenerationConfig.from_pretrained(
    pretrained_model_name='gpt2',
    cache_dir='../pretrained',
    max_new_tokens=100
)
text = "Replace me by any text you'd like."

generator = pipeline(task="text-generation", model=model, tokenizer=tokenizer)
output = generator(text, generation_config=cfg)
print(output)
