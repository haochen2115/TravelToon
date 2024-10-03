#coding:utf-8
import json 
import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxControlNetImg2ImgPipeline, FluxControlNetModel
from diffusers.utils import load_image
import cv2
import numpy as np

def convert_to_sketch(image_path, output_path):
    # 读取图片
    img = cv2.imread(image_path)
    # 转换为灰度图
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # 反转灰度图像
    inverted_gray = 255 - gray
    # 应用高斯模糊
    blurred = cv2.GaussianBlur(inverted_gray, (21, 21), 0)
    # 反转模糊图像
    inverted_blurred = 255 - blurred
    # 创建素描效果
    sketch = cv2.divide(gray, inverted_blurred, scale=256.0)
    cv2.imwrite(output_path, sketch)

config = json.load(open("config.json","r"))
flux_model_path = config['flux_model_path']
control_model_path = config['control_model_path']
sources = config['sources']
pipe1 = FluxPipeline.from_pretrained(flux_model_path, torch_dtype=torch.bfloat16)
pipe2 = FluxImg2ImgPipeline.from_pretrained(flux_model_path, torch_dtype=torch.bfloat16)
controlnet = FluxControlNetModel.from_pretrained(control_model_path, torch_dtype=torch.bfloat16)
pipe3 = FluxControlNetImg2ImgPipeline.from_pretrained(flux_model_path, controlnet=controlnet, torch_dtype=torch.bfloat16)
pipe1 = pipe1.to("cuda:0")
pipe2 = pipe2.to("cuda:1")
pipe3 = pipe3.to("cuda:2")

for source in sources:
    prompt = source['text']
    photo = source['photo']
    title = source['title']
    # Step 1: Text + Image To Image
    if(photo == ""):
        # Without image，pure prompt
        image = pipe1(
            prompt=prompt,
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
        ).images[0]
    else:
        input_image = f"./source/{photo}.png"
        # With image
        init_image = load_image(input_image).resize((1024, 1024))
        image = pipe2(
            prompt=prompt,
            image=init_image, 
            strength=0.95, 
            height=1024,
            width=1024,
            guidance_scale=3.5,
            num_inference_steps=50,
            max_sequence_length=512,
        ).images[0]
    image.save(f"./result/{title}_mid.png")
    convert_to_sketch(f"./result/{title}_mid.png", f"./result/{title}_mid_sketch.png")

    # Step 3: Image Style Transfer
    # Transfer to template style
    control_image = load_image(f"./result/{title}_mid_sketch.png")
    init_image = load_image("./source/init.png")
    image = pipe3(
        prompt=prompt,
        image=init_image,
        control_image=control_image,
        controlnet_conditioning_scale=0.6,
        strength=0.7,
        num_inference_steps=50,
        guidance_scale=3.5,
    ).images[0]
    image.save(f"./result/{title}_final.png")
    break

print("🎉 Please check your final result in ./result")