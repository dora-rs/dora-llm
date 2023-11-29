from sentence_transformers import SentenceTransformer
from sentence_transformers import util

from tqdm import tqdm
import torch
import os
import sys
import inspect


## Get all python files path in given directory
def get_all_functions(path):
    functions = []
    paths = []
    for root, dirs, files in os.walk(path):
        for file in files:
            if file.endswith(".py"):
                path = os.path.join(root, file)
                with open(path, "r", encoding="utf8") as f:
                    ## add file folder to system path
                    sys.path.append(root)
                    ## import module from path
                    # try:
                    # module = __import__(file[:-3])
                    ## get all functions in module
                    # for name, data in module.__dict__.items():
                    # if inspect.isfunction(data):
                    # ## Get function source code
                    # print(data)
                    # source = inspect.getsource(data)
                    # functions.append(source)
                    # paths.append(path)
                    # if inspect.isclass(data) and name == "Operator":
                    # ## Get function source code
                    # print(data)
                    # source = inspect.getsource(data)
                    # functions.append(source)
                    # paths.append(path)
                    # except Exception as err:
                    #     print(err)
                    #     pass
                    functions.append(f.read())
                    paths.append(file)
    return functions, paths


def search(
    query_embedding, corpus_embeddings, paths, sources, k=1, file_extension=None
):
    # TODO: filtering by file extension
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(k, len(cos_scores)), sorted=True)
    out = []
    for score, idx in zip(top_results[0], top_results[1]):
        out.append((score, paths[idx], sources[idx]))
    return out


model = SentenceTransformer("BAAI/bge-large-en-v1.5")
# model = SentenceTransformer(
# "krlvi/sentence-msmarco-bert-base-dot-v5-nlpl-code_search_net"
# )
# model = SentenceTransformer("embaas/sentence-transformers-e5-large-v2")


path = "/home/peter/Documents/work/dora-drives"
files, paths = get_all_functions(path)
sentence_embeddings = model.encode(files)

prompt = "What is ?"
query_embeddings = model.encode([prompt])

output = search(query_embeddings, sentence_embeddings, paths, files)


### ---


from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

model_name_or_path = "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
# Load model
model = AutoAWQForCausalLM.from_quantized(
    model_name_or_path, fuse_layers=True, trust_remote_code=False, safetensors=True
)


def query(system_message, prompt):
    prompt_template = f"""<|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    """
    token_input = tokenizer(prompt_template, return_tensors="pt").input_ids.cuda()
    # Generate output
    generation_output = model.generate(
        token_input,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_new_tokens=512,
    )
    # Get the tokens from the output, decode them, print them
    token_output = generation_output[0]
    text_output = tokenizer.decode(token_output)
    print("LLM output: ", text_output)


query(
    "You're a Python code expert. ",
    output[0][2] + "\n" + prompt,
)


from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer

# model_name_or_path = "TheBloke/Phind-CodeLlama-34B-v2-AWQ"
# model = AutoAWQForCausalLM.from_quantized(
# model_name_or_path, fuse_layers=True, trust_remote_code=False, safetensors=True
# )
# tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
# query("You're a Python code expert. ", "Print hello world with python")
# Load model

from pypdf import PdfReader

reader = PdfReader("ICLP_2014_Multiple-shots-on-SPDs-–additional-tests.pdf")
texts = ""
for page in reader.pages:
    texts += page.extract_text() + "\n"

paragraphs = texts.split("\n \n \n \n")

import torch
from awq import AutoAWQForCausalLM
from transformers import AutoTokenizer
from sentence_transformers import SentenceTransformer
from sentence_transformers import util

transformer_model = SentenceTransformer("BAAI/bge-large-en-v1.5")
model_name_or_path = "TheBloke/OpenHermes-2.5-Mistral-7B-AWQ"

# Load tokenizer
tokenizer = AutoTokenizer.from_pretrained(model_name_or_path, trust_remote_code=False)
# Load model
model = AutoAWQForCausalLM.from_quantized(
    model_name_or_path, fuse_layers=True, trust_remote_code=False, safetensors=True
)


def search(query_embedding, corpus_embeddings, sources, k=1, file_extension=None):
    # TODO: filtering by file extension
    cos_scores = util.cos_sim(query_embedding, corpus_embeddings)[0]
    top_results = torch.topk(cos_scores, k=min(k, len(cos_scores)), sorted=True)
    out = []
    for score, idx in zip(top_results[0], top_results[1]):
        out.append((score, sources[idx]))
    return out


def query(system_message, prompt):
    prompt_template = f"""<|im_start|>system
    {system_message}<|im_end|>
    <|im_start|>user
    {prompt}<|im_end|>
    <|im_start|>assistant
    """
    token_input = tokenizer(prompt_template, return_tensors="pt").input_ids.cuda()
    # Generate output
    generation_output = model.generate(
        token_input,
        do_sample=True,
        temperature=0.7,
        top_p=0.95,
        top_k=40,
        max_new_tokens=512,
    )
    # Get the tokens from the output, decode them, print them
    token_output = generation_output[0]
    text_output = tokenizer.decode(token_output)
    print("LLM output: ", text_output)


prompt = "What does this text explains us?"
query(
    "Help me explain this",
    texts + "\n" + prompt,
)


sentence_embeddings = transformer_model.encode(paragraphs)

del transformer_model
prompt = "What is a 10-pulse generator?"
query_embeddings = transformer_model.encode([prompt])
text = """
There are many evidences [1], [2], [3] (see also tests 
performed by Matt Darveniza from one side and Rick Gumley 
from  another  side  presented  in  previous  IEC  meetings)  that  
multiple shots can create problem to varistors even with 
magnitude  much  lower  than  the  maximum  capability  of  the  
varistor.  The  time  interval  for  multiple  strokes  is  typically  
around  30  ms  to  100  ms.  Previous  tests  have  shown  that  
varistors  that  withstand  many  tens  of  kA  can  only  handle  a  
few kA when repetitive strokes are applied. 
The  measured  lightning  current  shows  that  lightning  is  a  
continuous  process  with  multiple  pulses  [1].  When  lightning  
strikes  a  line  or  a  lightning  protection  system,  impulses  are  
injected  in  the  equipotential  bonding  SPD  at  the  entrance  of  
the installation. Same occurs for induced surges on the 
incoming  line.  Typical  surge  to  considers  are  in  the  range  of  
1 kA  to  15  kA  with  an  8/20  wave  for  induced  events  and  a  
waveshape  near  10/350  could  also  be  used  in  case  of  direct  
strike.
"""
output = search(query_embeddings, sentence_embeddings, paragraphs)
query(
    "Help me explain this in french",
    text + "\n" + prompt,
)
