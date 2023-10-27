#These nodes were made using code from the Deforum extension for A1111 webui
#You can find the project here: https://github.com/deforum-art/sd-webui-deforum
import comfy
import numexpr
import torch
import numpy as np
import pandas as pd
import re
import json


from .ScheduleFuncs import check_is_number, interpolate_prompts, interpolate_prompts_SDXL, PoolAnimConditioning, interpolate_string, addWeighted, reverseConcatenation
from .BatchFuncs import interpolate_prompt_series, BatchPoolAnimConditioning, BatchInterpolatePromptsSDXL #, BatchGLIGENConditioning
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
            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),
            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,})},# "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "app_text": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            }}
    
    RETURN_TYPES = ("CONDITIONING", )
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/ScheduleNodes"

    def animate(self, text, max_frames, current_frame, clip, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''):
        inputText = str("{" + text + "}")
        animation_prompts = json.loads(inputText.strip())
        cur_prompt, nxt_prompt, weight = interpolate_prompts(animation_prompts, max_frames, current_frame, pre_text, app_text, pw_a, pw_b, pw_c, pw_d)
        c = PoolAnimConditioning(cur_prompt, nxt_prompt, weight, clip,)
        return (c,)

class BatchPromptSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                             "clip": ("CLIP",),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": False, }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": False, }),  # "forceInput": True}),
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

    CATEGORY = "FizzNodes/BatchScheduleNodes"

    def animate(self, text, max_frames, clip, pw_a, pw_b, pw_c, pw_d, pre_text='', app_text=''):

        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)

        animation_prompts = json.loads(inputText.strip())
        print(animation_prompts)
        cur_prompt, nxt_prompt, weight = interpolate_prompt_series(animation_prompts, max_frames, pre_text,
        app_text, pw_a, pw_b, pw_c, pw_d)
        c = BatchPoolAnimConditioning(cur_prompt, nxt_prompt, weight, clip, )
        return (c,)

class BatchPromptScheduleLatentInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                             "clip": ("CLIP",),
                             "num_latents": ("LATENT", ),},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": False, }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": False, }),  # "forceInput": True}),
                             "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, }),
                             # "forceInput": True }),
                             }}

    RETURN_TYPES = ("CONDITIONING", "LATENT", )
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/BatchScheduleNodes"

    def animate(self, text, num_latents, clip, pw_a, pw_b, pw_c, pw_d, pre_text='', app_text=''):
        num_elements = sum([tensor.numel() for tensor in num_latents.values()])
        max_frames = num_elements

        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)

        animation_prompts = json.loads(inputText.strip())
        print(animation_prompts)
        cur_prompt, nxt_prompt, weight = interpolate_prompt_series(animation_prompts, max_frames, pre_text,
        app_text, pw_a, pw_b, pw_c, pw_d)
        c = BatchPoolAnimConditioning(cur_prompt, nxt_prompt, weight, clip, )
        return (c, num_latents, )
class StringSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),
                             "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0, })},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": False, }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": False, }),  # "forceInput": True}),
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

    CATEGORY = "FizzNodes/ScheduleNodes"

    def animate(self, text, max_frames, current_frame, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''):
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
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0})},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": False, }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": False, }),  # "forceInput": True}),
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

    CATEGORY = "FizzNodes/BatchScheduleNodes"

    def animate(self, text, max_frames, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''):
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        animation_prompts = json.loads(inputText.strip())
        cur_prompt_series, nxt_prompt_series, weight_series = interpolate_prompt_series(animation_prompts, max_frames, pre_text,
                                                             app_text, pw_a, pw_b, pw_c, pw_d)
        #c = PoolAnimConditioning(cur_prompt, nxt_prompt, weight, clip, )
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
            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),},
            "optional": {"pre_text_G": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "app_text_G": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "pre_text_L": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "app_text_L": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/BatchScheduleNodes"

    def animate(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, app_text_G, app_text_L, pre_text_G, pre_text_L, max_frames, pw_a, pw_b, pw_c, pw_d):
        inputTextG = str("{" + text_g + "}")
        inputTextL = str("{" + text_l + "}")
        inputTextG = re.sub(r',\s*}', '}', inputTextG)
        inputTextL = re.sub(r',\s*}', '}', inputTextL)
        animation_promptsG = json.loads(inputTextG.strip())
        animation_promptsL = json.loads(inputTextL.strip())
        return (BatchInterpolatePromptsSDXL(animation_promptsG, animation_promptsL, max_frames, clip,  app_text_G, app_text_L, pre_text_G, pre_text_L, pw_a, pw_b, pw_c, pw_d, width, height, crop_w, crop_h, target_width, target_height, ),)

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
            "num_latents": ("Latent", ),},
            "optional": {"pre_text_G": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "app_text_G": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "pre_text_L": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "app_text_L": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
             }}
    RETURN_TYPES = ("CONDITIONING", "LATENT",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/BatchScheduleNodes"

    def animate(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, app_text_G, app_text_L, pre_text_G, pre_text_L, num_latents, pw_a, pw_b, pw_c, pw_d):
        num_elements = sum([tensor.numel() for tensor in num_latents.values()])
        max_frames = num_elements
        inputTextG = str("{" + text_g + "}")
        inputTextL = str("{" + text_l + "}")
        inputTextG = re.sub(r',\s*}', '}', inputTextG)
        inputTextL = re.sub(r',\s*}', '}', inputTextL)
        animation_promptsG = json.loads(inputTextG.strip())
        animation_promptsL = json.loads(inputTextL.strip())
        return (BatchInterpolatePromptsSDXL(animation_promptsG, animation_promptsL, max_frames, clip,  app_text_G, app_text_L, pre_text_G, pre_text_L, pw_a, pw_b, pw_c, pw_d, width, height, crop_w, crop_h, target_width, target_height, ), num_latents, )

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
            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),
            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0})},# "forceInput": True}),},
            "optional": {"pre_text_G": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "app_text_G": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "pre_text_L": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "app_text_L": ("STRING", {"multiline": False,}),# "forceInput": True}),
            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}), #"forceInput": True }),
             }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/ScheduleNodes"

    def animate(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, app_text_G, app_text_L, pre_text_G, pre_text_L, max_frames, current_frame, pw_a, pw_b, pw_c, pw_d):
        inputTextG = str("{" + text_g + "}")
        inputTextL = str("{" + text_l + "}")
        inputTextG = re.sub(r',\s*}', '}', inputTextG)
        inputTextL = re.sub(r',\s*}', '}', inputTextL)
        animation_promptsG = json.loads(inputTextG.strip())
        animation_promptsL = json.loads(inputTextL.strip())
        return (interpolate_prompts_SDXL(animation_promptsG, animation_promptsL, max_frames, current_frame, clip,  app_text_G, app_text_L, pre_text_G, pre_text_L, pw_a, pw_b, pw_c, pw_d, width, height, crop_w, crop_h, target_width, target_height, ),)

# This node schedules the prompt using separate nodes as the keyframes.
# The values in the prompt are evaluated in NodeFlowEnd.
class PromptScheduleNodeFlow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),                           
                            "num_frames": ("INT", {"default": 24.0, "min": 0.0, "max": 9999.0, "step": 1.0}),},                           
               "optional":  {"in_text": ("STRING", {"multiline": False, }), # "forceInput": True}),
                             "max_frames": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,})}} # "forceInput": True}),}}
    
    RETURN_TYPES = ("INT","STRING",)

    FUNCTION = "addString"

    CATEGORY = "FizzNodes/ScheduleNodes"

    def addString(self, text, in_text = '', max_frames = 0, num_frames = 0,):
        
        if max_frames == 0:
            new_text = str("\"" + str(max_frames) + "\": \"" + text + "\"")
        else:
            new_text = str(in_text + "\n" + ",\"" + str(max_frames) + "\": \"" + text + "\"")
        
        new_max = num_frames + max_frames
        return (new_max, new_text,)

#Last node in the Node Flow for evaluating the json produced by the above node.
class PromptScheduleNodeFlowEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": False, "forceInput": True}), 
                            "clip": ("CLIP", ),
                            "max_frames": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}), #"forceInput": True}),
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),}, #"forceInput": True}),},
               "optional": {"pre_text": ("STRING", {"multiline": False, }),#"forceInput": True}),
                            "app_text": ("STRING", {"multiline": False, }),#"forceInput": True}),
                            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1,}),# "forceInput": True}),
                            }}
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/ScheduleNodes"

    def animate(self, text, max_frames, current_frame, clip, pw_a = 0, pw_b = 0, pw_c = 0, pw_d = 0, pre_text = '', app_text = ''):
        if text[-1] == ",":
            text = text[:-1]
        if text[0] == ",":
            text = text[:0]
        inputText = str("{"+text+"}") #format the input so it's valid json
        animation_prompts = json.loads(inputText.strip())
        return (interpolate_prompts(animation_prompts, max_frames, current_frame, clip, pre_text, app_text, pw_a, pw_b, pw_c, pw_d, ),) #return a conditioning value   

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
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),},
                # "forceInput": True}),},
                "optional": {"pre_text": ("STRING", {"multiline": False, }),  # "forceInput": True}),
                             "app_text": ("STRING", {"multiline": False, }),  # "forceInput": True}),
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

    CATEGORY = "FizzNodes/BatchScheduleNodes"

    def animate(self, conditioning_to, clip, gligen_textbox_model, text, width, height, x, y, max_frames, pw_a, pw_b, pw_c, pw_d, pre_text='', app_text=''):
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        animation_prompts = json.loads(inputText.strip())
        cur_series, nxt_series, weight_series = interpolate_prompt_series(animation_prompts, max_frames, pre_text, app_text, pw_a, pw_b, pw_c, pw_d)
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
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),
                             "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0,}),# "forceInput": True}),
                             }}
    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/ScheduleNodes"
    
    def animate(self, text, max_frames, current_frame,):
        t = get_inbetweens(parse_key_frames(text, max_frames), max_frames)
        cFrame = current_frame
        return (t[cFrame],int(t[cFrame]),)

class BatchValueSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultValue}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),
                             }}

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/BatchScheduleNodes"

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

    CATEGORY = "FizzNodes/BatchScheduleNodes"

    def animate(self, text, num_latents, ):
        num_elements = sum([tensor.numel() for tensor in num_latents.values()])
        max_frames = num_elements
        t = batch_get_inbetweens(batch_parse_key_frames(text, max_frames), max_frames)
        return (t, list(map(int,t)), num_latents, )