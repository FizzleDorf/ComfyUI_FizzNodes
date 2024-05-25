#These nodes were made using code from the Deforum extension for A1111 webui
#You can find the project here: https://github.com/deforum-art/sd-webui-deforum

import numexpr
import torch
import torch.nn.functional as F
import numpy as np
import pandas as pd
import re
import json

#functions used by PromptSchedule nodes

#This Settings class is mainly used to reduce clutter and keep things relatively
#organized. It is multi-purpose for both regular clip encoding and SDXL encoding
#The value schedule doesn't have as many arguments so I didn't bother doing the
#same for that.
class ScheduleSettings:
    def __init__(
            self,
            text_g: str,
            pre_text_G: str,
            app_text_G: str,
            text_L: str,
            pre_text_L: str,
            app_text_L: str,
            max_frames: int,
            current_frame: int,
            print_output: bool,
            pw_a: float,
            pw_b: float,
            pw_c: float,
            pw_d: float,
            start_frame: int,
            end_frame:int,
            width: int,
            height: int,
            crop_w: int,
            crop_h: int,
            target_width: int,
            target_height: int,
    ):
        self.text_g=text_g
        self.pre_text_G=pre_text_G
        self.app_text_G=app_text_G
        self.text_l=text_L
        self.pre_text_L=pre_text_L
        self.app_text_L=app_text_L
        self.max_frames=max_frames
        self.current_frame=current_frame
        self.print_output=print_output
        self.pw_a=pw_a
        self.pw_b=pw_b
        self.pw_c=pw_c
        self.pw_d=pw_d
        self.start_frame=start_frame
        self.end_frame=end_frame
        self.width=width
        self.height=height
        self.crop_w=crop_w
        self.crop_h=crop_h
        self.target_width=target_width
        self.target_height=target_height

    def set_sync_option(self, sync_option: bool):
        self.sync_context_to_pe = sync_option

#Addweighted function from Comfyui
def addWeighted(conditioning_to, conditioning_from, conditioning_to_strength, max_size=0):
    out = []

    if len(conditioning_from) > 1:
        print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

    cond_from = conditioning_from[0][0]
    pooled_output_from = conditioning_from[0][1].get("pooled_output", None)

    for i in range(len(conditioning_to)):
        t1 = conditioning_to[i][0]
        pooled_output_to = conditioning_to[i][1].get("pooled_output", pooled_output_from)
        if max_size == 0:
            max_size = max(t1.shape[1], cond_from.shape[1])
        t0, max_size = pad_with_zeros(cond_from, max_size)
        t1, max_size = pad_with_zeros(t1, t0.shape[1])  # Padding t1 to match max_size
        t0, max_size = pad_with_zeros(t0, t1.shape[1])

        tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
        t_to = conditioning_to[i][1].copy()

        t_to["pooled_output"] = pooled_output_from
        n = [tw, t_to]
        out.append(n)

    return out


def pad_with_zeros(tensor, target_length):
    current_length = tensor.shape[1]

    if current_length < target_length:
        # Calculate the required padding length
        pad_length = target_length - current_length

        # Calculate padding on both sides to maintain the tensor's original shape
        left_pad = pad_length // 2
        right_pad = pad_length - left_pad

        # Pad the tensor along the second dimension
        tensor = F.pad(tensor, (0, 0, left_pad, right_pad))

    return tensor, target_length

def process_input_text(text: str) -> dict:
    input_text = text.replace('\n', '')
    input_text = "{" + input_text + "}"
    input_text = re.sub(r',\s*}', '}', input_text)
    animation_prompts = json.loads(input_text.strip())
    return animation_prompts

def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)

def parse_weight(match, frame=0, max_frames=0) -> float: #calculate weight steps for in-betweens
        w_raw = match.group("weight")
        max_f = max_frames  # this line has to be left intact as it's in use by numexpr even though it looks like it doesn't
        if w_raw is None:
            return 1
        if check_is_number(w_raw):
            return float(w_raw)
        else:
            t = frame
            if len(w_raw) < 3:
                print('the value inside `-characters cannot represent a math function')
                return 1
            return float(numexpr.evaluate(w_raw[1:-1]))

def PoolAnimConditioning(cur_prompt, nxt_prompt, weight, clip):  
    if str(cur_prompt) == str(nxt_prompt):
        tokens = clip.tokenize(str(cur_prompt))
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    if weight == 1:
        tokens = clip.tokenize(str(cur_prompt))
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]

    if weight == 0:
        tokens = clip.tokenize(str(nxt_prompt))
        cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
        return [[cond, {"pooled_output": pooled}]]
    else:
        tokens = clip.tokenize(str(nxt_prompt))
        cond_from, pooled_from = clip.encode_from_tokens(tokens, return_pooled=True)
        tokens = clip.tokenize(str(cur_prompt))
        cond_to, pooled_to = clip.encode_from_tokens(tokens, return_pooled=True)
        return addWeighted([[cond_to, {"pooled_output": pooled_to}]], [[cond_from, {"pooled_output": pooled_from}]], weight)

def SDXLencode(g, l, settings:ScheduleSettings, clip):
    tokens = clip.tokenize(g)
    tokens["l"] = clip.tokenize(l)["l"]
    if len(tokens["l"]) != len(tokens["g"]):
        empty = clip.tokenize("")
        while len(tokens["l"]) < len(tokens["g"]):
            tokens["l"] += empty["l"]
        while len(tokens["l"]) > len(tokens["g"]):
            tokens["g"] += empty["g"]
    cond, pooled = clip.encode_from_tokens(tokens, return_pooled=True)
    return [[cond, {
        "pooled_output": pooled,
        "width": settings.width,
        "height": settings.height,
        "crop_w": settings.crop_w,
        "crop_h": settings.crop_h,
        "target_width": settings.target_width,
        "target_height": settings.target_height
    }]]