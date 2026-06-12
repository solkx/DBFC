from diffusers import CogView4Pipeline
import torch
import re
import json
import os
from tqdm import tqdm
def txt2json_mner(data):
    findata = []
    temp = []
    sent = []
    ents = {}
    img = ""
    for line in data:
        line = line.strip()
        if not line:
            if temp:
                ents[" ".join(temp[1:])] = temp[0]
                temp = []
            findata.append({
                "tokens": sent,
                "entity": ents,
                "img": img
            })
            sent = []
            ents = {}
            continue
        if "IMGID" in line:
            img = line.split(":")[-1]
            continue
        word = line.split("	")[0]
        label = line.split("	")[-1]
        if "-" in label:
            ent_t = label.split("-")[-1]
            label = label.split("-")[0]
            if ent_t == "OTHER":
                ent_t = "MISC"
        else:
            ent_t = ""
        sent.append(word)
        if label == "B":
            if temp:
                ents[" ".join(temp[1:])] = temp[0]
                temp = []
            temp.extend([ent_t, word])
        elif label == "I":
            if temp:
                temp.append(word)
        else:
            if temp:
                ents[" ".join(temp[1:])] = temp[0]
                temp = []
    if temp:
        ents[" ".join(temp[1:])] = temp[0]
        temp = []
    findata.append({
        "tokens": sent,
        "entity": ents,
        "img": img
    })
    return findata

pipe = CogView4Pipeline.from_pretrained("./CogView4-6B", torch_dtype=torch.bfloat16)

# Open it for reduce GPU memory usage
pipe.enable_model_cpu_offload()
pipe.vae.enable_slicing()
pipe.vae.enable_tiling()

for name in ["train", 'val', "test"]:
    with open(f'./twitter15_data/twitter2015/{name}.txt', 'r', encoding='utf-8') as f:
        train_data = txt2json_mner(f.read().split("\n"))
    for item in tqdm(train_data):
        img_id = item["img"]
        prompt = " ".join(item["tokens"])
        prompt = "Please generate an image containing the possible entities from the text. The text is" + " ".join(item["tokens"]) + "."
        image = pipe(
            prompt=prompt,
            guidance_scale=3.5,
            num_images_per_prompt=1,
            num_inference_steps=50,
            width=512,
            height=512,
        ).images[0]
        
        image.save(f"./{img_id}.png")
