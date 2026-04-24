import argparse
from pathlib import Path
import math
import json
from torch.utils.data import DataLoader
import torch
from transformers import AutoTokenizer, AutoModel, AutoConfig
from tqdm import tqdm

from multiview_dataset import MultiviewDataset


device = "cuda" if torch.cuda.is_available() else "cpu"


def split_model(model_path):
    device_map = {}
    world_size = torch.cuda.device_count()
    config = AutoConfig.from_pretrained(model_path, trust_remote_code=True)
    num_layers = config.llm_config.num_hidden_layers
    # Since the first GPU will be used for ViT, treat it as half a GPU.
    num_layers_per_gpu = math.ceil(num_layers / (world_size - 0.5))
    num_layers_per_gpu = [num_layers_per_gpu] * world_size
    num_layers_per_gpu[0] = math.ceil(num_layers_per_gpu[0] * 0.5)
    layer_cnt = 0
    for i, num_layer in enumerate(num_layers_per_gpu):
        for j in range(num_layer):
            device_map[f"language_model.model.layers.{layer_cnt}"] = i
            layer_cnt += 1
    device_map["vision_model"] = 0
    device_map["mlp1"] = 0
    device_map["language_model.model.tok_embeddings"] = 0
    device_map["language_model.model.embed_tokens"] = 0
    device_map["language_model.output"] = 0
    device_map["language_model.model.norm"] = 0
    device_map["language_model.model.rotary_emb"] = 0
    device_map["language_model.lm_head"] = 0
    device_map[f"language_model.model.layers.{num_layers - 1}"] = 0

    return device_map


path = "OpenGVLab/InternVL3-14B"

front_image_system_prompt = """
<image>\n
You are a highly skilled image captioning assistant. Your task is to generate accurate and concised
captions for a image with given textual descriptions. 
You are always presented with an image of a front facing camera mounted on a car. 
"""
image_examples_prompt = """
\n
Here are some examples of good image captioning practices:
* The ego vehicle is moving straight at a high speed with deceleration. There is no traffic light in the scene. The car is driving in a tunnel. The car is driving on a highway. What the driver of ego vehicle should be careful is that there is no traffic light in the scene, which means they need to rely on road signs and their own judgment to determine when it is safe to proceed. Additionally, the road appears to be wet or damp, as indicated by the reflective surface, which could increase the risk of hydroplaning if the vehicle were to encounter water on the road.\n\nThe presence of other vehicles ahead suggests that the ego vehicle should maintain a safe distance and be prepared to react to any sudden movements from the vehicles in front. The road appears to be relatively narrow with guardrails on both sides, so the ego vehicle should also be cautious of potential obstacles such as pedestrians, animals, or debris that might be on the road.\n\nIn summary, while the ego vehicle is moving at a high speed, the lack of traffic lights and the wet road conditions pose some risks that the driver should be aware of and take precautions against.", "risk": "that there is no traffic light in the scene, which means they need to rely on road signs and their own judgment to determine when it is safe to proceed. Additionally, the road appears to be wet or damp, as indicated by the reflective surface, which could increase the risk of hydroplaning if the vehicle were to encounter water on the road.\n\nThe presence of other vehicles ahead suggests that the ego vehicle should maintain a safe distance and be prepared to react to any sudden movements from the vehicles in front. The road appears to be relatively narrow with guardrails on both sides, so the ego vehicle should also be cautious of potential obstacles such as pedestrians, animals, or debris that might be on the road.\n\nIn summary, while the ego vehicle is moving at a high speed, the lack of traffic lights and the wet road conditions pose some risks that the driver should be aware of and take precautions against. 
* The ego vehicle is moving straight at a high speed. There is no traffic light in the scene. It is cloudy. The car is driving on a highway. No pedestrians appear to be present. What the driver of ego vehicle should be careful is that the road ahead is clear and there is no oncoming traffic or obstacles. The weather condition is cloudy, which might affect visibility, so it's important for the driver to maintain a safe distance from other vehicles and to be prepared to react quickly if necessary. Additionally, the road appears to be relatively empty, but the driver should still remain vigilant as there could always be unexpected situations such as animals crossing the road or sudden changes in road conditions.\n\nIn terms of special risks, the high speed and acceleration of the ego vehicle can increase the risk of accidents if the driver loses control or encounters an obstacle suddenly. It's crucial for the driver to stay focused and maintain a safe speed, especially given the cloudy weather conditions that may reduce visibility
"""
image_instructions_prompt = """
\n
Follow the guidelines step-by-step to create good video captions:
* Identify first the time of day. Classify the time of day in one of the following: daytime, dusk/dawn, nighttime
* Identify second the weather condition. Classify the weather in one of the following conditions: sunny, overcast, rainy, snowy, foggy. If the road is wet, this might indicate rainy weather
* Describe the road type with the number of lanes and the lane markings separating different lanes.
* Describe all the characteristic details of the identified vehicle, like color, type (car, truck, bus, etc.), and any visible features.
* Describe the positions and movements of nearby vehicles, road signs, and any changes in traffic flow.
* You should ensure that the captions are clear, concise, and informative, providing a comprehensive understanding
of the image content without any ambiguity.
* Limit your answer to three sentences!
* Start the first sentence with 'An image of a front center camera'
"""

rear_image_system_prompt = """
<image>\n
You are a highly skilled image captioning assistant. Your task is to generate accurate and concised
captions for a image with given textual descriptions. 
You are always presented with an image of a rear facing camera mounted on a car. 
"""
front_image_instructions_prompt = """
* Start the first sentence with 'An image of a front center camera'
"""
rear_image_instructions_prompt = """
* Start the first sentence with 'An image of a rear center camera'
"""

front_image_prompt = (
    front_image_system_prompt
    + image_instructions_prompt
    + front_image_instructions_prompt
)
rear_image_prompt = (
    rear_image_system_prompt
    + image_instructions_prompt
    + rear_image_instructions_prompt
)


def generate_captions(dataloader, model_path):
    model = (
        AutoModel.from_pretrained(
            model_path,
            torch_dtype=torch.bfloat16,
            low_cpu_mem_usage=True,
            use_flash_attn=True,
            trust_remote_code=True,
        )
        .eval()
        .to(device)
    )

    tokenizer = AutoTokenizer.from_pretrained(
        path, trust_remote_code=True, use_fast=False
    )
    generation_config = dict(
        max_new_tokens=1024, do_sample=True, temperature=0.3, top_k=50
    )
    image_captions = {}

    for batch_idx, batch in enumerate(tqdm(dataloader, total=len(dataloader))):
        batch_size = batch["front_img"].size(0)
        num_patches_list = []
        pixel_values_list = []
        img_id_list = []
        for idx in range(batch_size):
            pixel_values_front = batch["front_img"][idx].to(torch.bfloat16).to(device)
            pixel_values_rear = batch["rear_img"][idx].to(torch.bfloat16).to(device)
            num_patches_list.append(pixel_values_front.size(0))
            num_patches_list.append(pixel_values_rear.size(0))
            pixel_values_list.append(pixel_values_front)
            pixel_values_list.append(pixel_values_rear)
            img_id_list.extend([batch["front_id"][idx], batch["rear_id"][idx]])

        pixel_values = torch.cat(pixel_values_list, dim=0)

        questions = [front_image_prompt, rear_image_prompt] * batch_size

        responses = model.batch_chat(
            tokenizer,
            pixel_values,
            num_patches_list=num_patches_list,
            questions=questions,
            generation_config=generation_config,
        )

        for img_id, response in zip(img_id_list, responses):
            image_captions.update({img_id: response})

    return image_captions


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--h5_file", help="Path to h5 file with images to caption")
    parser.add_argument(
        "--model", default="OpenGVLab/InternVL3-14B", help="Model to use for captioning"
    )
    parser.add_argument("--batch_size", default=4)
    args = parser.parse_args()

    h5_file = Path(args.h5_file)
    model_path = args.model
    batch_size = int(args.batch_size)

    ds = MultiviewDataset(h5_file, input_size=448, use_InternVL3=True)
    dataloader = DataLoader(dataset=ds, batch_size=batch_size, shuffle=False)

    captions_dict = generate_captions(dataloader, model_path)

    file_id = h5_file.stem
    with open(file_id + ".json", "w") as fh:
        json.dump(captions_dict, fh, indent=2)

    print(f"Captions successfully created for {file_id}")


if __name__ == "__main__":
    main()
