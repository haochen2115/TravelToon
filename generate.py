# coding:utf-8
import json
import torch
from diffusers import FluxPipeline, FluxImg2ImgPipeline, FluxControlNetImg2ImgPipeline, FluxControlNetModel
from diffusers.utils import load_image
from utils import convert_to_sketch, merge_png


def load_config(config_path="config.json"):
    """Load configuration from a JSON file."""
    with open(config_path, "r", encoding="utf-8") as config_file:
        return json.load(config_file)


def initialize_pipelines(flux_model_path, control_model_path, lora_model_path, lora_weight_name, lora_adapter_name):
    """Initialize pipelines and load LoRA weights."""
    pipe1 = FluxPipeline.from_pretrained(flux_model_path, torch_dtype=torch.bfloat16)
    pipe2 = FluxImg2ImgPipeline.from_pretrained(flux_model_path, torch_dtype=torch.bfloat16)
    controlnet = FluxControlNetModel.from_pretrained(control_model_path, torch_dtype=torch.bfloat16)
    pipe3 = FluxControlNetImg2ImgPipeline.from_pretrained(flux_model_path, controlnet=controlnet, torch_dtype=torch.bfloat16)

    for pipe in [pipe1, pipe2, pipe3]:
        pipe.load_lora_weights(lora_model_path, weight_name=lora_weight_name, adapter_name=lora_adapter_name)
        pipe.set_adapters(lora_adapter_name)

    return pipe1, pipe2, pipe3


def assign_pipelines_to_devices(pipe1, pipe2, pipe3):
    """Assign each pipeline to a specific GPU device."""
    pipe1 = pipe1.to("cuda:0")
    pipe2 = pipe2.to("cuda:1")
    pipe3 = pipe3.to("cuda:2")
    return pipe1, pipe2, pipe3


def process_source(source, pipe1, pipe2, pipe3):
    """Process a single source, generating images."""
    prompt = source['text']
    photo = source['photo']
    title = source['title']

    result_path_v1 = f"./result/{title}_V1.png"
    result_path_v2 = f"./result/{title}_V2.png"

    # If no photo is provided, generate using pipe1
    if not photo:
        image = pipe1(prompt=prompt, height=1024, width=1024, guidance_scale=3.5, num_inference_steps=50,
                      max_sequence_length=512, generator=torch.Generator("cpu").manual_seed(0)).images[0]
        image.save(result_path_v1)
        image.save(result_path_v2)
        raw_image_path = f"./source/0.png"
        sketch_image_path = f"./source/0.png"
    else:
        # Generate V1 image using pipe2
        input_image_path = f"./source/{photo}.png"
        init_image = load_image(input_image_path).resize((1024, 1024))

        image = pipe2(prompt=prompt, image=init_image, strength=0.95, height=1024, width=1024, guidance_scale=3.5,
                      num_inference_steps=50, max_sequence_length=512, generator=torch.Generator("cpu").manual_seed(0)).images[0]
        image.save(result_path_v1)

        # Convert to sketch and process using controlnet in pipe3
        sketch_image_path = f"./source/{photo}_sketch.png"
        convert_to_sketch(input_image_path, sketch_image_path)
        control_image = load_image(sketch_image_path)

        image = pipe3(prompt=prompt, image=init_image, control_image=control_image, controlnet_conditioning_scale=0.5,
                      strength=0.95, num_inference_steps=100, guidance_scale=3.5, generator=torch.Generator("cpu").manual_seed(0)).images[0]
        image.save(result_path_v2)

        raw_image_path = input_image_path

    return title, raw_image_path, sketch_image_path, result_path_v1, result_path_v2


def generate_images_for_sources(sources, pipe1, pipe2, pipe3):
    """Generate images for all sources."""
    titles, raw_image_paths, sketch_image_paths, image_paths_1, image_paths_2 = [], [], [], [], []

    for source in sources:
        title, raw_image_path, sketch_image_path, result_path_v1, result_path_v2 = process_source(source, pipe1, pipe2, pipe3)
        titles.append(title)
        raw_image_paths.append(raw_image_path)
        sketch_image_paths.append(sketch_image_path)
        image_paths_1.append(result_path_v1)
        image_paths_2.append(result_path_v2)

    return titles, raw_image_paths, sketch_image_paths, image_paths_1, image_paths_2


def merge_images(titles, raw_image_paths, sketch_image_paths, image_paths_1, image_paths_2, travel_topic):
    """Merge generated images into final PNGs."""
    merge_png(titles, raw_image_paths, travel_topic, f"./result/{travel_topic}_RAW.png")
    merge_png(titles, sketch_image_paths, travel_topic, f"./result/{travel_topic}_SKETCH.png")
    merge_png(titles, image_paths_1, travel_topic, f"./result/{travel_topic}_V1.png")
    merge_png(titles, image_paths_2, travel_topic, f"./result/{travel_topic}_V2.png")


def main():
    # Load config
    config = load_config()

    # Initialize pipelines and assign devices
    pipe1, pipe2, pipe3 = initialize_pipelines(
        flux_model_path=config['flux_model_path'],
        control_model_path=config['control_model_path'],
        lora_model_path=config['lora_model_path'],
        lora_weight_name=config['lora_weight_name'],
        lora_adapter_name=config['lora_adapter_name']
    )
    pipe1, pipe2, pipe3 = assign_pipelines_to_devices(pipe1, pipe2, pipe3)

    # Generate images for sources
    titles, raw_image_paths, sketch_image_paths, image_paths_1, image_paths_2 = generate_images_for_sources(
        sources=config['sources'], pipe1=pipe1, pipe2=pipe2, pipe3=pipe3
    )

    # Merge images into final outputs
    merge_images(titles, raw_image_paths, sketch_image_paths, image_paths_1, image_paths_2, config['travel_topic'])

    print("🎉 Please check your final result in ./result")


if __name__ == "__main__":
    main()
