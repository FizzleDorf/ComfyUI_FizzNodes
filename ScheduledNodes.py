#These nodes were made using code from the Deforum extension for A1111 webui
#You can find the project here: https://github.com/deforum-art/sd-webui-deforum
import comfy
import numexpr
import torch
import numpy as np
import pandas as pd
import re
import json

from .ScheduleFuncs import *
from .BatchFuncs import *
from .ValueFuncs import *
from .ScheduleTypes import *

#Max resolution value for Gligen area calculation.
MAX_RESOLUTION=8192

#Default prompt for the prompt schedules.
defaultPrompt=""""0" :"",
"11" :"",
"23" :"",
"35" :"",
"47" :"",
"59" :"",
"71" :"",
"83" :"",
"95" :"",
"107" :"",
"119" :""
"""
#Default prompt for the value schedules.
defaultValue="""0:(0),
11:(0),
23:(0),
35:(0),
47:(0),
59:(0),
71:(0),
83:(0),
95:(0),
107:(0),
119:(0)
"""

#This node parses the user's formatted prompt,
#sequences the current prompt,next prompt, and
#conditioning strength, evaluates expressions in
#the prompts, and then returns either current,
#next or averaged conditioning.
class PromptSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default":defaultPrompt}),
                    "clip": ("CLIP", ),
                    "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                    "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0, "forceInput": True }),
                    "print_output":("BOOLEAN", {"default": False,}),
                },
                "optional": {"pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"
    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, text, max_frames, print_output, current_frame,clip, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''
    ):
        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=max_frames,
            current_frame=current_frame,
            print_output=print_output,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=0,
            end_frame=0,
            width=None,
            height=None,
            crop_w=None,
            crop_h=None,
            target_width=None,
            target_height=None,
        )
        return prompt_schedule(settings,clip)

# This node parses the user's formatted prompt,
# sequences the current prompt,next prompt, and
# conditioning strength, evaluates expressions in
# the prompts, and then returns a batch of
# conditionings with the schedule applied.
class BatchPromptSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                    "clip": ("CLIP",),
                    "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                    "print_output":("BOOLEAN", {"default": False}),
                },
                "optional": {
                    "pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "start_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "display": "start_frame(print_only)", }),
                    "end_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "display": "end_frame(print_only)",}),
                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, max_frames, print_output, clip, start_frame, end_frame, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''
    ):
        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=max_frames,
            current_frame=None,
            print_output=print_output,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=start_frame,
            end_frame=end_frame,
            width=None,
            height=None,
            crop_w=None,
            crop_h=None,
            target_width=None,
            target_height=None,
        )
        return batch_prompt_schedule(settings, clip)


# This node parses the user's formatted prompt,
# sequences the current prompt,next prompt, and
# conditioning strength, evaluates expressions in
# the prompts, and then returns a batch of
# conditionings with the schedule applied.
# This alternate batch node takes a latent as
# an input instead of max_frames
class BatchPromptScheduleLatentInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                    "clip": ("CLIP",),
                    "num_latents": ("LATENT", ),
                    "print_output":("BOOLEAN", {"default": False}),
                },
                "optional": {"pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "start_frame": ("INT", {"default": 0.0, "min": 0, "max": 9999, "step": 1, "display": "start_frame(print_only)", }),
                    "end_frame": ("INT", {"default": 0, "min": 0, "max": 9999, "step": 1, "display": "end_frame(print_only)", }),
                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",) #"CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG", "INPUT_LATENTS", ) #"POS_CUR", "NEG_CUR", "POS_NXT", "NEG_NXT",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, num_latents, print_output, clip, start_frame, end_frame, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text=''
    ):
        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=sum(tensor.size(0) for tensor in num_latents.values()),
            current_frame=None,
            print_output=print_output,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=start_frame,
            end_frame=end_frame,
            width=None,
            height=None,
            crop_w=None,
            crop_h=None,
            target_width=None,
            target_height=None,
        )
        return batch_prompt_schedule_latentInput(settings,clip, num_latents)

# This node prepares the strings and calculates
# the numexpr expressions. It returns a single
# string at the current_frame input.
class StringSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                     "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                     "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0, }),
                     "print_output":("BOOLEAN", {"default": False}),},
                "optional": {"pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                      "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                      "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                      "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                      "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                      "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, text, max_frames, current_frame, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text='', print_output = False ):
        settings = ScheduleSettings(
        text_g = text,
        pre_text_G = pre_text,
        app_text_G = app_text,
        text_L = None,
        pre_text_L = None,
        app_text_L = None,
        max_frames = max_frames,
        current_frame = current_frame,
        print_output = print_output,
        pw_a = pw_a,
        pw_b = pw_b,
        pw_c = pw_c,
        pw_d = pw_d,
        start_frame = 0,
        end_frame=0,
        width = None,
        height = None,
        crop_w = None,
        crop_h = None,
        target_width = None,
        target_height = None,
        )
        return string_schedule(settings)


# This node prepares the strings and calculates
# the numexpr expressions. It returns a batch of
# strings.
class BatchStringSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {
                    "text": ("STRING", {"multiline": True, "default": defaultPrompt}),
                    "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                    "print_output": ("BOOLEAN", {"default": False}),
            },
                "optional": {
                    "pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                }
        }

    RETURN_TYPES = ("STRING", "STRING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, max_frames,  pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='', app_text='',
                print_output=False):
        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=max_frames,
            current_frame=None,
            print_output=print_output,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=0,
            end_frame=0,
            width=None,
            height=None,
            crop_w=None,
            crop_h=None,
            target_width=None,
            target_height=None,
        )
        return batch_string_schedule(settings)

# Same as the regular node just for SDXL
# clips instead. the G and L clip can be
# scheduled separately before tokenization,
# goes through the same add_weighted process
# and returns the current, next or averaged
# conditioning.
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
                "text_g": ("STRING", {"multiline": True, }), "clip": ("CLIP", ),
                "text_l": ("STRING", {"multiline": True, }), "clip": ("CLIP", ),
                "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0}),
                "print_output":("BOOLEAN", {"default": False})
        },
            "optional": {
                "pre_text_G": ("STRING", {"multiline": True, "forceInput": True}),
                "app_text_G": ("STRING", {"multiline": True, "forceInput": True}),
                "pre_text_L": ("STRING", {"multiline": True, "forceInput": True}),
                "app_text_L": ("STRING", {"multiline": True, "forceInput": True}),
                "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
            }
        }
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, app_text_G, app_text_L, pre_text_G, pre_text_L, max_frames, current_frame, print_output, pw_a, pw_b, pw_c, pw_d):
        settings = ScheduleSettings(
            text_g=text_g,
            pre_text_G=pre_text_G,
            app_text_G=app_text_G,
            text_L=text_l,
            pre_text_L=pre_text_L,
            app_text_L=app_text_L,
            max_frames=max_frames,
            current_frame=current_frame,
            print_output=print_output,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=0,
            end_frame=0,
            width=width,
            height=height,
            crop_w=crop_w,
            crop_h=crop_h,
            target_width=target_width,
            target_height=target_height,
        )
        return prompt_schedule_SDXL(settings,clip)

# Same as the regular node just for SDXL
# clips instead. the G and L clip can be
# scheduled separately before tokenization,
# goes through the same add_weighted process
# and returns a batch of conditionings.
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
                    "text_g": ("STRING", {"multiline": True, }), "clip": ("CLIP", ),
                    "text_l": ("STRING", {"multiline": True, }), "clip": ("CLIP", ),
                    "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                    "print_output":("BOOLEAN", {"default": False}),
            },
                "optional": {
                    "pre_text_G": ("STRING", {"multiline": True, "forceInput": True}),
                    "app_text_G": ("STRING", {"multiline": True, "forceInput": True}),
                    "pre_text_L": ("STRING", {"multiline": True, "forceInput": True}),
                    "app_text_L": ("STRING", {"multiline": True, "forceInput": True}),
                    "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                    "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
            }
        }

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING",)# "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG", "POS_CUR", "NEG_CUR", "POS_NXT", "NEG_NXT",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, app_text_G, app_text_L, pre_text_G, pre_text_L, max_frames, print_output, pw_a=0, pw_b=0, pw_c=0, pw_d=0):
        settings = ScheduleSettings(
            text_g=text_g,
            pre_text_G=pre_text_G,
            app_text_G=app_text_G,
            text_L=text_l,
            pre_text_L=pre_text_L,
            app_text_L=app_text_L,
            max_frames=max_frames,
            current_frame=None,
            print_output=print_output,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=0,
            end_frame=0,
            width=width,
            height=height,
            crop_w=crop_w,
            crop_h=crop_h,
            target_width=target_width,
            target_height=target_height,
        )
        return batch_prompt_schedule_SDXL(settings, clip)

# Same as the regular node just for SDXL
# clips instead. the G and L clip can be
# scheduled separately before tokenization,
# goes through the same add_weighted process
# and returns a batch of conditionings. The
# max_size is input by the number of latents
# in the input.
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
                "text_g": ("STRING", {"multiline": True, }), "clip": ("CLIP", ),
                "text_l": ("STRING", {"multiline": True, }), "clip": ("CLIP", ),
                "num_latents": ("LATENT", ),
                "print_output":("BOOLEAN", {"default": False}),
        },
            "optional": {
                "pre_text_G": ("STRING", {"multiline": True, "forceInput": True}),
                "app_text_G": ("STRING", {"multiline": True, "forceInput": True}),
                "pre_text_L": ("STRING", {"multiline": True, "forceInput": True}),
                "app_text_L": ("STRING", {"multiline": True, "forceInput": True}),
                "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
             }
        }
    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "LATENT",)# "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG", "POS_CUR", "NEG_CUR", "POS_NXT", "NEG_NXT",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, clip, width, height, crop_w, crop_h, target_width, target_height, text_g, text_l, app_text_G, app_text_L, pre_text_G, pre_text_L, num_latents, print_output, pw_a, pw_b, pw_c, pw_d):
        settings = ScheduleSettings(
            text_g=text_g,
            pre_text_G=pre_text_G,
            app_text_G=app_text_G,
            text_L=text_l,
            pre_text_L=pre_text_L,
            app_text_L=app_text_L,
            max_frames=sum(tensor.size(0) for tensor in num_latents.values()),
            current_frame=None,
            print_output=print_output,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=0,
            end_frame=0,
            width=width,
            height=height,
            crop_w=crop_w,
            crop_h=crop_h,
            target_width=target_width,
            target_height=target_height,
        )
        return batch_prompt_schedule_SDXL_latentInput(settings, clip, num_latents)



# This node schedules the prompt using separate nodes as the keyframes.
# The values in the prompt are evaluated in NodeFlowEnd.
class PromptScheduleNodeFlow:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),                           
                            "num_frames": ("INT", {"default": 24.0, "min": 0.0, "max": 9999.0, "step": 1.0}),},
               "optional":  {"in_text": ("STRING", {"multiline": False, }), # "forceInput": True}),
                             "max_frames": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,})}}
    
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
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0, "forceInput": True}),},
               "optional": {"pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                            "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                            }}
    RETURN_TYPES = ("CONDITIONING","CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, text, max_frames, print_output, current_frame, clip, pw_a = 0, pw_b = 0, pw_c = 0, pw_d = 0, pre_text = '', app_text = ''):
        if text[-1] == ",":
            text = text[:-1]
        if text[0] == ",":
            text = text[:0]
        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=max_frames,
            current_frame=current_frame,
            print_output=print_output,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=0,
            end_frame=0,
            width=None,
            height=None,
            crop_w=None,
            crop_h=None,
            target_width=None,
            target_height=None,
        )
        return prompt_schedule(settings, clip)

#same as the other node end except it returns a batch
class BatchPromptScheduleNodeFlowEnd:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": False, "forceInput": True}),
                            "clip": ("CLIP", ),
                            "max_frames": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0,}),
                            "print_output": ("BOOLEAN", {"default": False}),
                            },
               "optional": {"pre_text": ("STRING", {"multiline": False, "forceInput": True}),
                            "app_text": ("STRING", {"multiline": False, "forceInput": True}),
                            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True}),
                            }}

    RETURN_TYPES = ("CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING", "CONDITIONING",)
    RETURN_NAMES = ("POS", "NEG", "POS_CUR", "NEG_CUR", "POS_NXT", "NEG_NXT",)

    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, max_frames, start_frame, end_frame, print_output, clip, pw_a=0, pw_b=0, pw_c=0, pw_d=0, pre_text='',
                app_text=''):
        if text[-1] == ",":
            text = text[:-1]
        if text[0] == ",":
            text = text[:0]
        settings = ScheduleSettings(
            text_g=text,
            pre_text_G=pre_text,
            app_text_G=app_text,
            text_L=None,
            pre_text_L=None,
            app_text_L=None,
            max_frames=max_frames,
            current_frame=None,
            print_output=print_output,
            pw_a=pw_a,
            pw_b=pw_b,
            pw_c=pw_c,
            pw_d=pw_d,
            start_frame=start_frame,
            end_frame=end_frame,
            width=None,
            height=None,
            crop_w=None,
            crop_h=None,
            target_width=None,
            target_height=None,
        )
        return batch_prompt_schedule(settings, clip)

# WIP, requires some hijacking but otherwise
# applies every scheduled gligen bound box to
# a batch of latents with the scheduled
# conditionings
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
                "optional": {"pre_text": ("STRING", {"multiline": True, "forceInput": True}),
                             "app_text": ("STRING", {"multiline": True, "forceInput": True}),
                             "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                             "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                             "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
                             "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1, "forceInput": True }),
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
                             "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0, "forceInput": True}),
                             "print_output": ("BOOLEAN", {"default": False})}}
    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"
    
    def animate(self, text, max_frames, current_frame, print_output):
        current_frame = current_frame % max_frames
        t = get_inbetweens(parse_key_frames(text, max_frames), max_frames)
        if (print_output is True):
            print("ValueSchedule: ",current_frame,"\n","current_frame: ",current_frame)
        return (t[current_frame],int(t[current_frame]),)

class BatchValueSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultValue}),
                            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                            "print_output": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("FLOAT", "INT")
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, max_frames, print_output):
        t = batch_get_inbetweens(batch_parse_key_frames(text, max_frames), max_frames)
        if print_output is True:
            print("ValueSchedule: ", t)
        return (t, list(map(int,t)),)

class BatchValueScheduleLatentInput:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True, "default": defaultValue}),
                             "num_latents": ("LATENT", ),
                             "print_output": ("BOOLEAN", {"default": False})}}

    RETURN_TYPES = ("FLOAT", "INT", "LATENT", )
    FUNCTION = "animate"

    CATEGORY = "FizzNodes 📅🅕🅝/BatchScheduleNodes"

    def animate(self, text, num_latents, print_output):
        num_elements = sum(tensor.size(0) for tensor in num_latents.values())
        max_frames = num_elements
        t = batch_get_inbetweens(batch_parse_key_frames(text, max_frames), max_frames)
        if print_output is True:
            print("ValueSchedule: ", t)
        return (t, list(map(int,t)), num_latents, )

# Expects a Batch Value Schedule list input,
# it exports an image batch with images taken
# from an input image batch.
# Original code is from:
# ComfyUI-Image-Selector by SLAPaper
# https://github.com/SLAPaper/ComfyUI-Image-Selector
# licensed under Apache-2.0
class ImagesFromBatchSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {
            "required": {
                "images": ("IMAGE",),
                "text": ("STRING", {"multiline": True, "default":defaultPrompt}),
                "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 999999.0, "step": 1.0, }),
                "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 999999.0, "step": 1.0}),
                "print_output": ("BOOLEAN", {"default": False}),
            }
        }

    RETURN_TYPES = ("IMAGE",)
    FUNCTION = "animate"
    CATEGORY = "FizzNodes 📅🅕🅝/ScheduleNodes"

    def animate(self, images, text, current_frame, max_frames, print_output):
        inputText = str("{" + text + "}")
        inputText = re.sub(r',\s*}', '}', inputText)
        animation_prompts = json.loads(inputText.strip())
        pos_cur_prompt, pos_nxt_prompt, weight = interpolate_prompt_series(animation_prompts, max_frames, 0, "",
                                                                           "", 0,
                                                                           0, 0, 0, print_output)
        selImages = selectImages(images,pos_cur_prompt[current_frame])
        return selImages

def selectImages(images: torch.Tensor, selected_indexes: str):
    shape = images.shape
    len_first_dim = shape[0]

    selected_index: list[int] = []
    total_indexes: list[int] = list(range(len_first_dim))
    for s in selected_indexes.strip().split(','):
        try:
            if ":" in s:
                _li = s.strip().split(':', maxsplit=1)
                _start = _li[0]
                _end = _li[1]
                if _start and _end:
                    selected_index.extend(
                        total_indexes[int(_start) - 1:int(_end) - 1]
                    )
                elif _start:
                    selected_index.extend(
                        total_indexes[int(_start) - 1:]
                    )
                elif _end:
                    selected_index.extend(
                        total_indexes[:int(_end) - 1]
                    )
            else:
                x: int = int(s.strip()) - 1
                if x < len_first_dim:
                    selected_index.append(x)
        except:
            pass

    if selected_index:
        print(f"ImageSelector: selected: {len(selected_index)} images")
        return (images[selected_index], )

    print(f"ImageSelector: selected no images, passthrough")
    return images