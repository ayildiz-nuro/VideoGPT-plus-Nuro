from tqdm import tqdm
from videogpt_plus.conversation import conv_templates
from videogpt_plus.model.builder import load_pretrained_model
from videogpt_plus.mm_utils import tokenizer_image_token, get_model_name_from_path
from eval.vcgbench.inference.ddp import *
from eval.video_encoding import _get_rawvideo_dec
import traceback
import pandas as pd
import torch
import os

# Disable parameter resetting for specific layers
setattr(torch.nn.Linear, "reset_parameters", lambda self: None)
setattr(torch.nn.LayerNorm, "reset_parameters", lambda self: None)

def infer(model, tokenizer, image_processor, video_processor, video_path, qs):
    conv_mode = "default"
    stop_str = "<|end|>"
    temperature = 0.0

    # Process video if it exists
    if os.path.exists(video_path):
        video_frames, context_frames, slice_len = _get_rawvideo_dec(
            video_path,
            image_processor,
            video_processor,
            max_frames=NUM_FRAMES,
            image_resolution=224,
            num_video_frames=NUM_FRAMES,
            num_context_images=NUM_CONTEXT_IMAGES,
        )

    # Prepare query string with image tokens
    if model.config.mm_use_im_start_end:
        qs = (
            DEFAULT_IM_START_TOKEN
            + DEFAULT_IMAGE_TOKEN * slice_len
            + DEFAULT_IM_END_TOKEN
            + "\n"
            + qs
        )
    else:
        qs = DEFAULT_IMAGE_TOKEN * slice_len + "\n" + qs

    # Create conversation prompt
    conv = conv_templates[conv_mode].copy()
    conv.append_message(conv.roles[0], qs)
    conv.append_message(conv.roles[1], None)
    prompt = conv.get_prompt()

    input_ids = (
        tokenizer_image_token(
            prompt, tokenizer, IMAGE_TOKEN_INDEX, return_tensors="pt"
        )
        .unsqueeze(0)
        .cuda()
    )

    # Generate output
    with torch.inference_mode():
        output_ids = model.generate(
            input_ids,
            images=torch.stack(video_frames).half().cuda(),
            context_images=torch.stack(context_frames).half().cuda(),
            do_sample=True if temperature > 0 else False,
            temperature=temperature,
            top_p=None,
            num_beams=1,
            max_new_tokens=1024,
            use_cache=True,
        )

    # Validate output
    input_token_len = input_ids.shape[1]
    n_diff_input_output = (
        (input_ids != output_ids[:, :input_token_len]).sum().item()
    )
    if n_diff_input_output > 0:
        print(
            f"[Warning] {n_diff_input_output} output_ids are not the same as the input_ids"
        )

    # Decode and clean up output
    outputs = tokenizer.batch_decode(
        output_ids[:, input_token_len:], skip_special_tokens=True
    )[0]
    outputs = outputs.strip()
    if outputs.endswith(stop_str):
        outputs = outputs[: -len(stop_str)]
    outputs = outputs.strip()
    outputs = outputs.replace("<|end|>", "")
    outputs = outputs.strip()

    print(outputs)
    return outputs


def load_model(model_path, model_base):

    # Load pretrained model and tokenizer
    model_path = os.path.expanduser(model_path)
    model_name = get_model_name_from_path(model_path)
    tokenizer, model, image_processor, context_len = load_pretrained_model(
        model_path, model_base, model_name
    )

    # Configure model for multimodal use
    mm_use_im_start_end = getattr(model.config, "mm_use_im_start_end", False)
    mm_use_im_patch_token = getattr(model.config, "mm_use_im_patch_token", True)
    if mm_use_im_patch_token:
        tokenizer.add_tokens([DEFAULT_IMAGE_PATCH_TOKEN], special_tokens=True)
    if mm_use_im_start_end:
        tokenizer.add_tokens(
            [DEFAULT_IM_START_TOKEN, DEFAULT_IM_END_TOKEN], special_tokens=True
        )
    model.resize_token_embeddings(len(tokenizer))

    # Load vision towers
    vision_tower = model.get_vision_tower()
    vision_tower.load_model(model.config.mm_vision_tower)
    video_processor = vision_tower.image_processor

    image_vision_tower = model.get_image_vision_tower()
    image_vision_tower.load_model()
    image_processor = image_vision_tower.image_processor

    # Move model to GPU
    model = model.to("cuda")
    return model, tokenizer, image_processor, video_processor


if __name__ == "__main__":
    model_path = "VideoGPT-plus_Phi3-mini-4k/vcgbench"
    model_base = "microsoft/Phi-3-mini-4k-instruct"

    # model_path = "VideoGPT-plus_LLaMA3-8B-8k/vcgbench"
    # model_base = "meta-llama/Meta-Llama-3-8B-Instruct"

    model, tokenizer, image_processor, video_processor = load_model(model_path, model_base)

    video_path = "/home/ayildiz/sample_videos/video_nexar_perf_trim.mov"
    # video_path = "/home/ayildiz/sample_videos/Event_ID_22583554_Front_cut.mp4"
    
    qs = """
    This is a footage from a dashcam from the ego vehicle. You need to explain the conflicts that occur in the video.
    Start by breaking the scene down into 3 main parts: 
    1. Description of road geometry. Static elements, and the environment. E.g. description of the traffic intersection, road markings, traffic lights, etc.
    2. Which agents are present in the scene, and what are they doing to cause a conflict? Be sure to mention their directions of travel from the perspective of the ego vehicle, and clarify whether their trajectories are parallel, orthogonal, or intersecting. Explain how their movements affect the potential conflict with the ego vehicle."
    3. What is the stage of the scenario progression? Each of these stages should capture agents' progress, intent, and interactions with the ego vehicle. Be very specific about the intent of the agents in the scene, and how they are causing a conflict with the ego vehicle.
    """

    outputs = infer(model, tokenizer, image_processor, video_processor, video_path, qs)

    import ipdb; ipdb.set_trace()


# ipython scripts/apply_delta.py -- --base-model-path llama_weights  --target-model-path LLaVA-Lightning-7B-v1-1 --delta-path liuhaotian/LLaVA-Lightning-7B-delta-v1-1