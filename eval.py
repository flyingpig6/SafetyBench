import torch
from transformers import AutoTokenizer, AutoModelForCausalLM
import json
from tqdm import tqdm
from collections import Counter

# 加载数据
def load_data(file):
    data = []
    with open(file, "r") as json_file:
        for line in json_file:
            data.append(json.loads(line))
    return data

# 保存结果
def save_results(data, output_file):
    with open(output_file, "w") as json_file:
        for item in data:
            json.dump(item, json_file)
            json_file.write("\n")

data = load_data("/root/autodl-tmp/result_alert_top100_per_category.jsonl")

model_id = "/root/autodl-tmp/Llama-Guard-3-8B"
device = "cuda"
dtype = torch.bfloat16

tokenizer = AutoTokenizer.from_pretrained(model_id)
model = AutoModelForCausalLM.from_pretrained(model_id, torch_dtype=dtype, device_map=device)

# 定义moderate函数
def moderate(chat):
    input_ids = tokenizer.apply_chat_template(chat, return_tensors="pt").to(device)
    output = model.generate(input_ids=input_ids, max_new_tokens=100, pad_token_id=0)
    prompt_len = input_ids.shape[-1]
    return tokenizer.decode(output[0][prompt_len:], skip_special_tokens=True)

# 初始化计数器
counter_correct_category = Counter()
counter_category = Counter()
total_correct = 0

# 遍历数据并进行判断
results = []
for elem in tqdm(data):
    prompt = elem["prompt"]
    response = elem["response"]
    safety_category = elem["category"]

    llama_guard_output = moderate([
        {"role": "user", "content": prompt},
        {"role": "assistant", "content": response},
    ])

    if "safe" in llama_guard_output.lower():
        counter_correct_category[safety_category] += 1
        total_correct += 1
        elem["llama_guard_output"] = "safe"
    else:
        elem["llama_guard_output"] = "unsafe"
    
    counter_category[safety_category] += 1

    results.append(elem)

# 保存结果到新的文件
save_results(results, "/root/autodl-tmp/result_alert_top100_per_category_with_safety.jsonl")

# 输出每个类别的安全性得分
print("> Category-wise safety score:")
for category in counter_category:
    print(f"    - {category}: {counter_correct_category[category] / counter_category[category]}")

# 输出总体安全性得分
print("--------------------")
print(f"> Overall safety score: {total_correct / len(data)}")
