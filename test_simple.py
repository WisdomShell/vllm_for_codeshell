from vllm import LLM, SamplingParams
import time
import random
import string
prompts = [
    "'## human:你是谁?<|endoftext|>## assistant:'",
]
sampling_params = SamplingParams(temperature=0.2, top_p=0.95, max_tokens=1024)
# llm = LLM(model="/nvme/share/checkpoints/codeshell/project/32k/ft_codeshell_7b_32k_20240113/iter_0000195/hf", trust_remote_code=True)
llm = LLM(model="/shd/zzr/models-ft/codeshell-ide-0116", trust_remote_code=True)

start = time.time()
outputs = llm.generate(prompts, sampling_params)
end = time.time()
total = end - start
# Print the outputs.

id_count = 0
for output in outputs:
    prompt = output.prompt
    id_count += len(output.outputs[0].token_ids)
    generated_text = output.outputs[0].text
    print(f"Prompt: {prompt!r}, Generated text: {generated_text!r}")

print(total, id_count, id_count/total)

