#These nodes were made using code from the Deforum extension for A1111 webui
#You can find the project here: https://github.com/deforum-art/sd-webui-deforum
import comfy
import numexpr
import torch
import numpy as np
import pandas as pd
import re
import json


from .ScheduleFuncs import (
    check_is_number, interpolate_prompts_SDXL, PoolAnimConditioning,
    interpolate_string, addWeighted, reverseConcatenation, split_weighted_subprompts
)
from .BatchFuncs import interpolate_prompt_series, BatchPoolAnimConditioning, BatchInterpolatePromptsSDXL, batch_split_weighted_subprompts #, BatchGLIGENConditioning
from .ValueFuncs import batch_get_inbetweens, batch_parse_key_frames, parse_key_frames, get_inbetweens, sanitize_value
#Max resolution value for Gligen area calculation.
MAX_RESOLUTION=8192

defaultPrompt=""""0" :"",
"12" :"",
"24" :"",
"36" :"",
"48" :"",
"60" :"",
"72" :"",
"84" :"",
"96" :"",
"108" :"",
"120" :""
"""

defaultValue="""0:(0),
12:(0),
24:(0),
36:(0),
48:(0),
60:(0),
72:(0),
84:(0),
96:(0),
108:(0),
120:(0)
"""

#This node parses the user's formatted prompt,
#sequences the current prompt,next prompt, and
#conditioning strength, evaluates expressions in
#the prompts, and then returns either current,
#next or averaged conditioning.
class PromptSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default":defaultPrompt}),
            "clip": ("CLIP", ),
            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,}),
            "print_output":("BOOLEAN", {"default": False}),},# "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": True,}),# "forceInput": True}),
            "app_text": ("STRING", {"multiline": True,}),# "forceInput": True}),
            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"
    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, text, max_frames, print_output, current_frame, clip, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''):
        current_frame = current_frame % max_frames
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        animation_prompts = json.loads(inputText.strip())
        start_frame = 0
        pos, neg = batch_split_weighted_subprompts(animation_prompts, pre_text, app_text)

        pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_series(pos, max_frames, start_frame, pre_text, app_text, pw_a,
                                                                           pw_b, pw_c, pw_d, print_output)
        pc = PoolAnimConditioning(pos_cur_prompt[current_frame], pos_nxt_prompt[current_frame], weight[current_frame], clip)

        neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_series(neg, max_frames, start_frame, pre_text, app_text, pw_a,
                                                                           pw_b, pw_c, pw_d, print_output)
        nc = PoolAnimConditioning(neg_cur_prompt[current_frame], neg_nxt_prompt[current_frame], weight[current_frame], clip)

        return (pc, nc,)

class BatchPromptSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                             "clip": ("CLIP",),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                             "print_output":("BOOLEAN", {"default": False}),},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": True, "default": "PRE" }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": True, "default": "APP" }),  # "forceInput": True}),
                             "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, }),
                             "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, max_frames, print_output, clip, start_frame, pw_a, pw_b, pw_c, pw_d, pre_text='', app_text=''):
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        max_frames += start_frame
        animation_prompts = json.loads(inputText.strip())
        pos, neg = batch_split_weighted_subprompts(animation_prompts, pre_text, app_text)

        pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_series(pos, max_frames, start_frame, pre_text, app_text, pw_a, pw_b, pw_c, pw_d, print_output)
        pc = BatchPoolAnimConditioning( pos_cur_prompt, pos_nxt_prompt, weight, clip,)

        neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_series(neg, max_frames, start_frame, pre_text, app_text, pw_a, pw_b, pw_c, pw_d, print_output)
        nc = BatchPoolAnimConditioning(neg_cur_prompt, neg_nxt_prompt, weight, clip, )

        return (pc, nc, )

class BatchPromptScheduleLatentInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                             "clip": ("CLIP",),
                             "num_latents": ("LATENT", ),
                             "print_output":("BOOLEAN", {"default": False}),},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": True, "default": "PRE", }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": True, "default": "APP", }),  # "forceInput": True}),
                             "start_frame": ("INT", {"default": 0.0, "min": 0, "max": 9999, "step": 1, }),
                             "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT", )
    RETURN_NAMES = ("POS", "NEG", "INPUT_LATENTS",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, num_latents, print_output, clip, start_frame, pw_a, pw_b, pw_c, pw_d, pre_text='', app_text=''):
        max_frames = sum(tensor.size(0) for tensor in num_latents.values())
        max_frames += start_frame
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)

        animation_prompts = json.loads(inputText.strip())
        pos, neg = batch_split_weighted_subprompts(animation_prompts, pre_text, app_text)

        pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_series(pos, max_frames, start_frame, pre_text,
                                                                           app_text, pw_a, pw_b, pw_c, pw_d,
                                                                           print_output)
        pc = BatchPoolAnimConditioning(pos_cur_prompt, pos_nxt_prompt, weight, clip, )

        neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_series(neg, max_frames, start_frame, pre_text,
                                                                           app_text, pw_a, pw_b, pw_c, pw_d,
                                                                           print_output)
        nc = BatchPoolAnimConditioning(neg_cur_prompt, neg_nxt_prompt, weight, clip, )

        return (pc, nc, num_latents,)
class StringSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                             "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0, })},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": True, "default": "PRE",  }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": True, "default": "APP", }),  # "forceInput": True}),
                             "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, text, max_frames, current_frame, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''):
        current_frame = current_frame % max_frames
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        animation_prompts = json.loads(inputText.strip())
        cur_prompt = interpolate_string(animation_prompts, max_frames, current_frame, pre_text,
                                                             app_text, pw_a, pw_b, pw_c, pw_d)
        #c = PoolAnimConditioning(cur_prompt, nxt_prompt, weight, clip, )
        return (cur_prompt,)

class PromptScheduleSDXLRefiner:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "ascore": ("FLOAT", {"default": 6.0, "min": 0.0, "max": 1000.0, "step": 0.01}),
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text": ("STRING", {"multiline": True, "default":defaultPrompt}), "clip": ("CLIP", ),
            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "encode"

    CATEGORY = "advanced/conditioning"

    def encode(self, clip, ascore, width, height, text):
        tokens = clip.tokenize(text)
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return ([[cond, {"pooled_output": pooled, "aesthetic_score": ascore, "width": width,"height": height}]], )

class BatchStringSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": True, "default": "PRE", }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": True, "default": "APP", }),  # "forceInput": True}),
                             "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             }}

    RETURN_TYPES = ("STRING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, max_frames, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''):
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        start_frame = 0
        animation_prompts = json.loads(inputText.strip())
        cur_prompt_series, nxt_prompt_series, weight_series = interpolate_prompt_series(animation_prompts, max_frames, start_frame, pre_text,
                                                             app_text, pw_a, pw_b, pw_c, pw_d)
        return (cur_prompt_series,)

class BatchPromptScheduleEncodeSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text_g": ("STRING", {"multiline": True, "default": "CLIP_G"}), "clip": ("CLIP", ),
            "text_l": ("STRING", {"multiline": True, "default": "CLIP_L"}), "clip": ("CLIP", ),
            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
            "print_output":("BOOLEAN", {"default": False}),},
            "optional": {"pre_text_G": ("STRING", {"multiline": True, "default": "PRE_G"}),# "forceInput": True}),
            "app_text_G": ("STRING", {"multiline": True, "default": "APP_G"}),# "forceInput": True}),
            "pre_text_L": ("STRING", {"multiline": True, "default": "PRE_L"}),# "forceInput": True}),
            "app_text_L": ("STRING", {"multiline": True, "default": "APP_L"}),# "forceInput": True}),
            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, app_text_G, app_text_L, pre_text_G, pre_text_L, max_frames, print_output, pw_a, pw_b, pw_c, pw_d):
        inputTextG = str("{" + text_g + "}")
        inputTextL = str("{" + text_l + "}")
        inputTextG = re.sub(r',\s*}', '}', inputTextG)
        inputTextL = re.sub(r',\s*}', '}', inputTextL)
        animation_promptsG = json.loads(inputTextG.strip())
        animation_promptsL = json.loads(inputTextL.strip())
        return (BatchInterpolatePromptsSDXL(animation_promptsG, animation_promptsL, max_frames, clip,  app_text_G, app_text_L, pre_text_G, pre_text_L, pw_a, pw_b, pw_c, pw_d, width, height, crop_w, crop_h, target_width, target_height, print_output,),)

class BatchPromptScheduleEncodeSDXLLatentInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text_g": ("STRING", {"multiline": True, "default": "CLIP_G"}), "clip": ("CLIP", ),
            "text_l": ("STRING", {"multiline": True, "default": "CLIP_L"}), "clip": ("CLIP", ),
            "num_latents": ("LATENT", ),
            "print_output":("BOOLEAN", {"default": False}),},
            "optional": {"pre_text_G": ("STRING", {"multiline": True, "default": "PRE_G",}),# "forceInput": True}),
            "app_text_G": ("STRING", {"multiline": True, "default": "APP_G",}),# "forceInput": True}),
            "pre_text_L": ("STRING", {"multiline": True, "default": "PRE_L",}),# "forceInput": True}),
            "app_text_L": ("STRING", {"multiline": True, "default": "APP_L",}),# "forceInput": True}),
            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
             }}
    RETURN_TYPES = ("CONDITIONING", "LATENT",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, app_text_G, app_text_L, pre_text_G, pre_text_L, num_latents, print_output, pw_a, pw_b, pw_c, pw_d):
        max_frames = sum(tensor.size(0) for tensor in num_latents.values())
        inputTextG = str("{" + text_g + "}")
        inputTextL = str("{" + text_l + "}")
        inputTextG = re.sub(r',\s*}', '}', inputTextG)
        inputTextL = re.sub(r',\s*}', '}', inputTextL)
        animation_promptsG = json.loads(inputTextG.strip())
        animation_promptsL = json.loads(inputTextL.strip())
        return (BatchInterpolatePromptsSDXL(animation_promptsG, animation_promptsL, max_frames, clip,  app_text_G, app_text_L, pre_text_G, pre_text_L, pw_a, pw_b, pw_c, pw_d, width, height, crop_w, crop_h, target_width, target_height, print_output, ), num_latents, )

class PromptScheduleEncodeSDXL:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
            "width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_w": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "crop_h": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION}),
            "target_width": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "target_height": ("INT", {"default": 1024.0, "min": 0, "max": MAX_RESOLUTION}),
            "text_g": ("STRING", {"multiline": True, "default": "CLIP_G"}), "clip": ("CLIP", ),
            "text_l": ("STRING", {"multiline": True, "default": "CLIP_L"}), "clip": ("CLIP", ),
            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0}),
            "print_output":("BOOLEAN", {"default": False})},
            "optional": {"pre_text_G": ("STRING", {"multiline": True, "default": "PRE_G",}),# "forceInput": True}),
            "app_text_G": ("STRING", {"multiline": True, "default": "APP_G",}),# "forceInput": True}),
            "pre_text_L": ("STRING", {"multiline": True, "default": "PRE_L",}),# "forceInput": True}),
            "app_text_L": ("STRING", {"multiline": True, "default": "APP_L",}),# "forceInput": True}),
            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, app_text_G, app_text_L, pre_text_G, pre_text_L, max_frames, current_frame, print_output, pw_a, pw_b, pw_c, pw_d):
        current_frame = current_frame % max_frames
        inputTextG = str("{" + text_g + "}")
        inputTextL = str("{" + text_l + "}")
        inputTextG = re.sub(r',\s*}', '}', inputTextG)
        inputTextL = re.sub(r',\s*}', '}', inputTextL)
        animation_promptsG = json.loads(inputTextG.strip())
        animation_promptsL = json.loads(inputTextL.strip())
        return (interpolate_prompts_SDXL(animation_promptsG, animation_promptsL, max_frames, current_frame, clip,  app_text_G, app_text_L, pre_text_G, pre_text_L, pw_a, pw_b, pw_c, pw_d, width, height, crop_w, crop_h, target_width, target_height, print_output,),)

# This node schedules the prompt using separate nodes as the keyframes.
# The values in the prompt are evaluated in NodeFlowEnd.
class PromptScheduleNodeFlow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),                           
                            "num_frames": ("INT", {"default": 24.0, "min": 0.0, "max": 9999.0, "step": 1.0}),},
               "optional":  {"in_text": ("STRING", {"multiline": False, }), # "forceInput": True}),
                             "max_frames": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,})}} # "forceInput": True}),}}
    
    RETURN_TYPES = ("INT","STRING",)
    FUNCTION = "addString"
    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def addString(self, text, in_text='', max_frames=0, num_frames=0):
        if in_text:
            # Remove trailing comma from in_text if it exists
            in_text = in_text.rstrip(',')

        new_max = num_frames + max_frames

        if max_frames == 0:
            # Construct a new JSON object with a single key-value pair
            new_text = in_text + (', ' if in_text else '') + f'"{max_frames}": "{text}"'
        else:
            # Construct a new JSON object with a single key-value pair
            new_text = in_text + (', ' if in_text else '') + f'"{new_max}": "{text}"'



        return (new_max, new_text,)


#Last node in the Node Flow for evaluating the json produced by the above node.
class PromptScheduleNodeFlowEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": False, "forceInput": True}), 
                            "clip": ("CLIP", ),
                            "max_frames": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,}),
                            "print_output": ("BOOLEAN", {"default": False}),
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,}),}, #"forceInput": True}),},
               "optional": {"pre_text": ("STRING", {"multiline": True, "default": "PRE", }),#"forceInput": True}),
                            "app_text": ("STRING", {"multiline": True, "default": "APP", }),#"forceInput": True}),
                            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            }}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, text, max_frames, print_output, current_frame, clip, pw_a = 0, pw_b = 0, pw_c = 0, pw_d = 0, pre_text = '', app_text = ''):
        current_frame = current_frame % max_frames
        if text[-1] == ",":
            text = text[:-1]
        if text[0] == ",":
            text = text[:0]
        start_frame = 0
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        animation_prompts = json.loads(inputText.strip())
        max_frames += start_frame
        pos, neg = batch_split_weighted_subprompts(animation_prompts, pre_text, app_text)

        pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_series(pos, max_frames, start_frame, pre_text, app_text, pw_a,
                                                                           pw_b, pw_c, pw_d, print_output)
        pc = PoolAnimConditioning(pos_cur_prompt[current_frame], pos_nxt_prompt[current_frame], weight[current_frame],
                                  clip, )

        neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_series(neg, max_frames, start_frame, pre_text, app_text, pw_a,
                                                                           pw_b, pw_c, pw_d, print_output)
        nc = PoolAnimConditioning(neg_cur_prompt[current_frame], neg_nxt_prompt[current_frame], weight[current_frame],
                                  clip, )

        return (pc, nc,)

class BatchPromptScheduleNodeFlowEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": False, "forceInput": True}),
                            "clip": ("CLIP", ),
                            "max_frames": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,}),
                            "print_output": ("BOOLEAN", {"default": False}),
                            },
               "optional": {"pre_text": ("STRING", {"multiline": False, }),#"forceInput": True}),
                            "app_text": ("STRING", {"multiline": False, }),#"forceInput": True}),
                            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            }}
    RETURN_TYPES = ("CONDITIONING",)

    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, max_frames, start_frame, print_output, clip, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', current_frame = 0,
                app_text=''):
        if text[-1] == ",":
            text = text[:-1]
        if text[0] == ",":
            text = text[:0]
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        animation_prompts = json.loads(inputText.strip())

        max_frames += start_frame

        pos, neg = batch_split_weighted_subprompts(animation_prompts, pre_text, app_text)

        pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_series(pos, max_frames, start_frame, pre_text, app_text, pw_a,
                                                                           pw_b, pw_c, pw_d, print_output)
        pc = BatchPoolAnimConditioning(pos_cur_prompt[current_frame], pos_nxt_prompt[current_frame], weight[current_frame],
                                  clip, )

        neg_cur_prompt, neg_nxt_prompt, weight = interpolate_prompt_series(neg, max_frames, start_frame, pre_text, app_text, pw_a,
                                                                           pw_b, pw_c, pw_d, print_output)
        nc = BatchPoolAnimConditioning(neg_cur_prompt[current_frame], neg_nxt_prompt[current_frame], weight[current_frame],
                                  clip, )

        return (pc, nc,)

class BatchGLIGENSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"conditioning_to": ("CONDITIONING",),
                             "clip": ("CLIP",),
                             "gligen_textbox_model": ("GLIGEN",),
                             "text": ("STRING", {"multiline": True, "default":defaultPrompt}),
                             "width": ("INT", {"default": 64, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
                             "height": ("INT", {"default": 64, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
                             "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                             "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                             "print_output":("BOOLEAN", {"default": False})},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": True, "default": "PRE", }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": True, "default": "APP", }),  # "forceInput": True}),
                             "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             }}

    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, conditioning_to, clip, gligen_textbox_model, text, width, height, x, y, max_frames, print_output, pw_a, pw_b, pw_c, pw_d, pre_text='', app_text=''):
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        animation_prompts = json.loads(inputText.strip())

        cur_series, nxt_series, weight_series = interpolate_prompt_series(animation_prompts, max_frames, pre_text, app_text, pw_a, pw_b, pw_c, pw_d, print_output)
        out = []
        for i in range(0, max_frames - 1):
            # Calculate changes in x and y here, based on your logic
            x_change = 8
            y_change = 0

            # Update x and y values
            x += x_change
            y += y_change
            print(x)
            print(y)
            out.append(self.append(conditioning_to, clip, gligen_textbox_model, pre_text, width, height, x, y))

        return (out,)

    def append(self, conditioning_to, clip, gligen_textbox_model, text, width, height, x, y):
        c = []
        cond, cond_pooled = clip.encode_from_tokens(clip.tokenize(text), return_pooled=True)
        for t in range(0, len(conditioning_to)):
            n = [conditioning_to[t][0], conditioning_to[t][1].copy()]
            position_params = [(cond_pooled, height // 8, width // 8, y // 8, x // 8)]
            prev = []
            if "gligen" in n[1]:
                prev = n[1]['gligen'][2]

            n[1]['gligen'] = ("position", gligen_textbox_model, prev + position_params)
            c.append(n)
        return c

#This node parses the user's test input into 
#interpolated floats. Expressions can be input 
#and evaluated.            
class ValueSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default":defaultValue}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                             "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,}),# "forceInput": True}),
                             }}
    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"
    
    def animate(self, text, max_frames, current_frame,):
        current_frame = current_frame % max_frames
        t = get_inbetweens(parse_key_frames(text, max_frames), max_frames)
        cFrame = current_frame
        return (t[cFrame],int(t[cFrame]),)

class BatchValueSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultValue}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                             }}

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, max_frames, ):
        t = batch_get_inbetweens(batch_parse_key_frames(text, max_frames), max_frames)
        return (t, list(map(int,t)),)

class BatchValueScheduleLatentInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultValue}),
                             "num_latents": ("LATENT", ),
                             }}

    RETURN_TYPES = ("FLOAT", "INT", "LATENT", )
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, num_latents, ):
        num_elements = sum(tensor.size(0) for tensor in num_latents.values())
        max_frames = num_elements
        t = batch_get_inbetweens(batch_parse_key_frames(text, max_frames), max_frames)
        return (t, list(map(int,t)), num_latents, )

# Expects a Batch Value Schedule list input, it exports an image batch with images taken from an input image batch
class ImageBatchFromValueSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "values": ("FLOAT", { "default": 1.0, "min": -1.0, "max": 1.0, "label": "values" }),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "animate"
    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, images, values):
        values = [values] * n if isinstance(values, float) else values
        min_value, max_value = min(values), max(values)
        i = [(x - min_value) / (max_value - min_value) * (images.shape[0] - 1) for x in values]
        return (images[i], )
