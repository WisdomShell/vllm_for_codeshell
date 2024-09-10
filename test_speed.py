import time
import torch
# from text_generation import Client
from openai import OpenAI

from transformers import AutoModelForCausalLM, AutoTokenizer

input_text = ""
with open("/shd/zzr/code_copilot/speed_test/test.txt", "r", encoding="utf-8") as f:
    for line in f.readlines():
        input_text += line

device = 'cuda' if torch.cuda.is_available() else 'cpu'
# model_path = '/nvme/share/checkpoints/codeshell/project/32k/ft_codeshell_7b_32k_20240108/iter_0000209/hf'
# model_path = '/shd/zzr/models/codellama-7b'
model_path = '/nvme/share/checkpoints/codeshell/project/32k/cpt_codeshell_7b_32k_20240107/iter_0000560/hf'
tokenizer = AutoTokenizer.from_pretrained(model_path)
# model = AutoModelForCausalLM.from_pretrained("/shd/zzr/models/codellama-7b", trust_remote_code=True, torch_dtype=torch.bfloat16).to(device)
inputs = tokenizer(input_text, return_tensors='pt').to(device)
input_text = tokenizer.decode(inputs.input_ids[0][:4000])

print(type(input_text))
inputs = tokenizer(input_text, return_tensors='pt').to(device)
print("original len(inputs.input_ids)", len(inputs.input_ids[0]))

# outputs = model.generate(**inputs, max_length=1024)
# print(tokenizer.decode(outputs[0]))

start_time = time.time()
print("start time:", start_time)

# client = Client("http://127.0.0.1:9123", timeout=1000)
# respone = client.generate(input_text, max_new_tokens=1024, temperature=1.0)
# respone = respone.generated_text

openai_api_key = "EMPTY"
openai_api_base = "http://localhost:8000/v1"
client = OpenAI(
    api_key=openai_api_key,
    base_url=openai_api_base,
)
respone = client.completions.create(model=model_path,
                                      prompt=input_text, max_tokens=1024, temperature=1.0)
respone = respone.choices[0].text
# print("Completion result:", respone)

end_time = time.time()
print("end time:", end_time)
inputs = tokenizer(respone, return_tensors='pt').to(device)
print("output len(inputs.input_ids)", len(inputs.input_ids[0]))

print("gen speed:", len(inputs.input_ids[0]) / (end_time - start_time))
