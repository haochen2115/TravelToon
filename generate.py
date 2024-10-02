import json 
import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxControlNetImg2ImgPipeline, FluxControlNetModel
from diffusers.utils import load_image

config = json.load(open("config.json","r"))
flux_model_path = config['flux_model_path']
control_model_path = config['control_model_path']
device = "cuda"
pipe1 = FluxPipeline.from_pretrained(flux_model_path, torch_dtype=torch.bfloat16)
pipe2 = FluxImg2ImgPipeline.from_pretrained(flux_model_path, torch_dtype=torch.bfloat16)
controlnet = FluxControlNetModel.from_pretrained(control_model_path, torch_dtype=torch.bfloat16)
pipe3 = FluxControlNetImg2ImgPipeline.from_pretrained(flux_model_path, controlnet=controlnet, torch_dtype=torch.bfloat16)
# pipe3.text_encoder.to(torch.float16)
# pipe3.controlnet.to(torch.float16)
pipe1 = pipe1.to(device)
pipe2 = pipe2.to(device)
pipe3 = pipe3.to(device)

for source in sources:
    prompt = source['text']
    # Step 1: Text + Image To Image
    if(source['photo'] == "")
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
        input_image = f"./source/{source['photo']}.jpeg" 
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
    image.save(f"./result/{source['photo']}_mid.jpeg")
    
    # Step 2: Image Style Transfer
    # Transfer to template style

    control_image = load_image("./source/template.jpeg")
    init_image = load_image(f"./result/{source['photo']}_mid.jpeg")
    image = pipe(
        prompt=prompt,
        image=init_image,
        control_image=control_image,
        controlnet_conditioning_scale=0.6,
        strength=0.7,
        num_inference_steps=2,
        guidance_scale=3.5,
    ).images[0]
    image.save(f"./result/{source['photo']}_final.jpeg")

    break