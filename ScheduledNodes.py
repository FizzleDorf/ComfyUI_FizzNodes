#These nodes were made using code from the Deforum extension for A1111 webui
#You can find the project here: https://github.com/deforum-art/sd-webui-deforum
import comfy
import numexpr
import torch
import numpy as np
import pandas as pd
import re
import json


from .ScheduleFuncs import check_is_number, interpolate_prompts, interpolate_prompts_SDXL, PoolAnimConditioning, interpolate_string
from .BatchFuncs import interpolate_prompt_series, BatchPoolAnimConditioning, BatchInterpolatePromptsSDXL
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
        animation_prompts = json.loads(inputText.strip())
        cur_prompt, nxt_prompt, weight = interpolate_prompt_series(animation_prompts, max_frames, pre_text,
        app_text, pw_a, pw_b, pw_c, pw_d)
        c = BatchPoolAnimConditioning(cur_prompt, nxt_prompt, weight, clip, )
        return (c,)

class BatchSDXLPromptSchedule:
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
        animation_prompts = json.loads(inputText.strip())
        cur_prompt, nxt_prompt, weight = interpolate_prompt_series(animation_prompts, max_frames, pre_text,
        app_text, pw_a, pw_b, pw_c, pw_d)
        c = BatchPoolAnimConditioning(cur_prompt, nxt_prompt, weight, clip, )
        return (c,)

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
        animation_promptsG = json.loads(inputTextG.strip())
        animation_promptsL = json.loads(inputTextL.strip())
        return (BatchInterpolatePromptsSDXL(animation_promptsG, animation_promptsL, max_frames, clip,  app_text_G, app_text_L, pre_text_G, pre_text_L, pw_a, pw_b, pw_c, pw_d, width, height, crop_w, crop_h, target_width, target_height, ),)

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
        t = self.get_inbetweens(self.parse_key_frames(text, max_frames), max_frames)
        cFrame = current_frame
        return (t[cFrame],int(t[cFrame]),)

    def sanitize_value(self, value):
        return value.replace("'","").replace('"',"").replace('(',"").replace(')',"")

    def get_inbetweens(self, key_frames, max_frames, integer=False, interp_method='Linear', is_single_string = False):
        key_frame_series = pd.Series([np.nan for a in range(max_frames)])
        max_f = max_frames -1 #needed for numexpr even though it doesn't look like it's in use.
        value_is_number = False
        for i in range(0, max_frames):
            if i in key_frames:
                value = key_frames[i]
                value_is_number = check_is_number(self.sanitize_value(value))
                if value_is_number: # if it's only a number, leave the rest for the default interpolation
                    key_frame_series[i] = self.sanitize_value(value)
            if not value_is_number:
                t = i
                # workaround for values formatted like 0:("I am test") //used for sampler schedules
                key_frame_series[i] = numexpr.evaluate(value) if not is_single_string else self.sanitize_value(value)
            elif is_single_string:# take previous string value and replicate it
                key_frame_series[i] = key_frame_series[i-1]
        key_frame_series = key_frame_series.astype(float) if not is_single_string else key_frame_series # as string
    
        if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
            interp_method = 'Quadratic'
        if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
            interp_method = 'Linear'
    
        key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
        key_frame_series[max_frames-1] = key_frame_series[key_frame_series.last_valid_index()]
        key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')
        
        if integer:
            return key_frame_series.astype(int)
        return key_frame_series
    
    def parse_key_frames(self, string, max_frames):
        # because math functions (i.e. sin(t)) can utilize brackets 
        # it extracts the value in form of some stuff
        # which has previously been enclosed with brackets and
        # with a comma or end of line existing after the closing one
        frames = dict()
        for match_object in string.split(","):
            frameParam = match_object.split(":")
            max_f = max_frames -1 #needed for numexpr even though it doesn't look like it's in use.
            frame = int(self.sanitize_value(frameParam[0])) if check_is_number(self.sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(frameParam[0].strip().replace("'","",1).replace('"',"",1)[::-1].replace("'","",1).replace('"',"",1)[::-1]))
            frames[frame] = frameParam[1].strip()
        if frames == {} and len(string) != 0:
            raise RuntimeError('Key Frame string not correctly formatted')
        return frames


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
        t = self.get_inbetweens(self.parse_key_frames(text, max_frames), max_frames)
        return (t, list(map(int,t)),)

    def sanitize_value(self, value):
        return value.replace("'","").replace('"',"").replace('(',"").replace(')',"")
    def get_inbetweens(self, key_frames, max_frames, integer=False, interp_method='Linear', is_single_string=False):
        key_frame_series = pd.Series([np.nan for a in range(max_frames)])
        max_f = max_frames - 1  # needed for numexpr even though it doesn't look like it's in use.
        value_is_number = False
        for i in range(0, max_frames):
            if i in key_frames:
                value = key_frames[i]
                value_is_number = check_is_number(self.sanitize_value(value))
                if value_is_number:  # if it's only a number, leave the rest for the default interpolation
                    key_frame_series[i] = self.sanitize_value(value)
            if not value_is_number:
                t = i
                # workaround for values formatted like 0:("I am test") //used for sampler schedules
                key_frame_series[i] = numexpr.evaluate(value) if not is_single_string else self.sanitize_value(value)
            elif is_single_string:  # take previous string value and replicate it
                key_frame_series[i] = key_frame_series[i - 1]
        key_frame_series = key_frame_series.astype(float) if not is_single_string else key_frame_series  # as string

        if interp_method == 'Cubic' and len(key_frames.items()) <= 3:
            interp_method = 'Quadratic'
        if interp_method == 'Quadratic' and len(key_frames.items()) <= 2:
            interp_method = 'Linear'

        key_frame_series[0] = key_frame_series[key_frame_series.first_valid_index()]
        key_frame_series[max_frames - 1] = key_frame_series[key_frame_series.last_valid_index()]
        key_frame_series = key_frame_series.interpolate(method=interp_method.lower(), limit_direction='both')

        if integer:
            return key_frame_series.astype(int)
        return key_frame_series

    def parse_key_frames(self, string, max_frames):
        # because math functions (i.e. sin(t)) can utilize brackets
        # it extracts the value in form of some stuff
        # which has previously been enclosed with brackets and
        # with a comma or end of line existing after the closing one
        frames = dict()
        for match_object in string.split(","):
            frameParam = match_object.split(":")
            max_f = max_frames - 1  # needed for numexpr even though it doesn't look like it's in use.
            frame = int(self.sanitize_value(frameParam[0])) if check_is_number(
                self.sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(
                frameParam[0].strip().replace("'", "", 1).replace('"', "", 1)[::-1].replace("'", "", 1).replace('"', "",
                                                                                                                1)[
                ::-1]))
            frames[frame] = frameParam[1].strip()
        if frames == {} and len(string) != 0:
            raise RuntimeError('Key Frame string not correctly formatted')
        return frames