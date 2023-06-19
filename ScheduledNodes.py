#These nodes were made using code from the Deforum extension for A1111 webui
#You can find the project here: https://github.com/deforum-art/sd-webui-deforum

import numexpr
import torch
import numpy as np
import pandas as pd
import re
import simplejson as json

#Max resolution value for Gligen area calculation.
MAX_RESOLUTION=8192

# used by both nodes
def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)

#functions used by PromptSchedule nodes

def addWeighted(conditioning_to, conditioning_from, conditioning_to_strength): #Addweighted function from Comfyui
        out = []

        if len(conditioning_from) > 1:
            print("Warning: ConditioningAverage conditioning_from contains more than 1 cond, only the first one will actually be applied to conditioning_to.")

        cond_from = conditioning_from[0][0]

        for i in range(len(conditioning_to)):
            t1 = conditioning_to[i][0]
            t0 = cond_from[:,:t1.shape[1]]
            if t0.shape[1] < t1.shape[1]:
                t0 = torch.cat([t0] + [torch.zeros((1, (t1.shape[1] - t0.shape[1]), t1.shape[2]))], dim=1)

            tw = torch.mul(t1, conditioning_to_strength) + torch.mul(t0, (1.0 - conditioning_to_strength))
            n = [tw, conditioning_to[i][1].copy()]
            out.append(n)
        return (out, )

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

def prepare_prompt(prompt_series, max_frames, frame_idx, prompt_weight_1 = 0, prompt_weight_2 = 0, prompt_weight_3 = 0, prompt_weight_4 = 0): #calculate expressions from the text input and return a string
        max_f = max_frames - 1
        pattern = r'`.*?`' #set so the expression will be read between two backticks (``)
        regex = re.compile(pattern)
        prompt_parsed = str(prompt_series)
        for match in regex.finditer(prompt_parsed):
            matched_string = match.group(0)
            parsed_string = matched_string.replace('t', f'{frame_idx}').replace("pw_a", f"prompt_weight_1").replace("pw_b", f"prompt_weight_2").replace("pw_c", f"prompt_weight_3").replace("pw_d", f"prompt_weight_4").replace("max_f", f"{max_f}").replace('`', '') #replace t, max_f and `` respectively
            parsed_value = numexpr.evaluate(parsed_string)
            prompt_parsed = prompt_parsed.replace(matched_string, str(parsed_value))
        return prompt_parsed.strip()

#This node parses the user's formatted prompt,
#sequences the current prompt,next prompt, and 
#conditioning strength, evalates expressions in 
#the prompts, and then returns either current, 
#next or averaged conditioning.
class PromptSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}), 
                            "clip": ("CLIP", ),
                            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0}),},
               "optional": {"pre_text": ("STRING", {"multiline": False}),
                            "app_text": ("STRING", {"multiline": False}),
                            "pw_a": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1}),
                            "pw_b": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1}),
                            "pw_c": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1}),
                            "pw_d": ("FLOAT", {"default": 0.0, "min": -9999.0, "max": 9999.0, "step": 0.1}),
                            }}
    
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/ScheduleNodes"

    def animate(self, text, max_frames, current_frame, clip, pw_a = 0, pw_b = 0, pw_c = 0, pw_d = 0, pre_text = '', app_text = ''):
        inputText = str("{"+text+"}") #format the input so it's valid json
        animation_prompts = json.loads(inputText.strip())
        return self.interpolate_prompts(animation_prompts, max_frames, current_frame, clip, pre_text, app_text, pw_a, pw_b, pw_c, pw_d) #return a conditioning value   
    
    def interpolate_prompts(self, animation_prompts, max_frames, current_frame, clip, pre_text, app_text, prompt_weight_1, prompt_weight_2, prompt_weight_3, prompt_weight_4): #parse the conditioning strength and determine in-betweens.
        #Get prompts sorted by keyframe
        max_f = max_frames #needed for numexpr even though it doesn't look like it's in use.
        parsed_animation_prompts = {}
        for key, value in animation_prompts.items():
            if check_is_number(key):  #default case 0:(1 + t %5), 30:(5-t%2)
                parsed_animation_prompts[key] = value
            else:  #math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                parsed_animation_prompts[int(numexpr.evaluate(key))] = value
        
        sorted_prompts = sorted(parsed_animation_prompts.items(), key=lambda item: int(item[0]))
        
        #Setup containers for interpolated prompts
        cur_prompt_series = pd.Series([np.nan for a in range(max_frames)])
        nxt_prompt_series = pd.Series([np.nan for a in range(max_frames)])

        #simple array for strength values
        weight_series = [np.nan] * max_frames

        #in case there is only one keyed promt, set all prompts to that prompt
        if len(sorted_prompts) - 1 == 0:
            for i in range(0, len(cur_prompt_series)-1):           
                current_prompt = sorted_prompts[0][1]           
                cur_prompt_series[i] = str(pre_text) + " " + str(current_prompt) + " " + str(app_text)
                nxt_prompt_series[i] = str(pre_text) + " " + str(current_prompt) + " " + str(app_text)

        # For every keyframe prompt except the last
        for i in range(0, len(sorted_prompts) - 1):
            # Get current and next keyframe
            current_key = int(sorted_prompts[i][0])
            next_key = int(sorted_prompts[i + 1][0])


            # Ensure there's no weird ordering issues or duplication in the animation prompts
            # (unlikely because we sort above, and the json parser will strip dupes)
            if current_key >= next_key:
                print(f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
                continue
    
            # Get current and next keyframes' positive and negative prompts (if any)
            current_prompt = sorted_prompts[i][1]
            next_prompt = sorted_prompts[i + 1][1]
            
            # Calculate how much to shift the weight from current to next prompt at each frame.
            weight_step = 1 / (next_key - current_key)
    
            for f in range(current_key, next_key):
                next_weight = weight_step * (f - current_key)
                current_weight = 1 - next_weight
                
                #add the appropriate prompts and weights to their respective containers.
                cur_prompt_series[f] = ''
                nxt_prompt_series[f] = ''
                weight_series[f] = 0.0

                cur_prompt_series[f] += (str(pre_text) + " " + str(current_prompt) + " " + str(app_text))
                nxt_prompt_series[f] += (str(pre_text) + " " + str(current_prompt) + " " + str(app_text))

                weight_series[f] += current_weight
        
        #Evaluate the current and next prompt's expressions
        cur_prompt_series[current_frame] = prepare_prompt(cur_prompt_series[current_frame], max_frames, current_frame, prompt_weight_1, prompt_weight_2, prompt_weight_3, prompt_weight_4)
        nxt_prompt_series[current_frame] = prepare_prompt(nxt_prompt_series[current_frame], max_frames, current_frame, prompt_weight_1, prompt_weight_2, prompt_weight_3, prompt_weight_4)       

        #Show the to/from prompts with evaluated expressions for transparency.
        print(" current prompt: ",cur_prompt_series[current_frame], "\n","next prompt: ", nxt_prompt_series[current_frame], "\n", "strength : ", weight_series[current_frame])

        #Output methods depending if the prompts are the same or if the current frame is a keyframe.
        #if it is an in-between frame and the prompts differ, compostable diffusion will be performed.
        if str(cur_prompt_series[current_frame]) == str(nxt_prompt_series[current_frame]):
            return ([[clip.encode(str(cur_prompt_series[current_frame])), {}]], )
        if weight_series[current_frame] == 1:
            return ([[clip.encode(str(cur_prompt_series[current_frame])), {}]], ) #Will probably never trigger but I'm paranoid
        if weight_series[current_frame] == 0:
            return ([[clip.encode(str(nxt_prompt_series[current_frame])), {}]], )
        else:
            return addWeighted(list([[clip.encode(str(cur_prompt_series[current_frame])), {}]], ), list([[clip.encode(str(nxt_prompt_series[current_frame])), {}]], ), weight_series[current_frame])

#This node is the same as above except it takes 
#a conditioning value and a GLIGEN model to work 
#with GLIGEN nodes. Gligen code adapted from Comfyui.

class PromptScheduleGLIGEN:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}), 
                            "clip": ("CLIP", ),
                            "conditioning_to":("CONDITIONING",),
                            "gligen_textbox_model": ("GLIGEN", ),
                            "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),
                            "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0}),
                            "width": ("INT", {"default": 64, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
                            "height": ("INT", {"default": 64, "min": 8, "max": MAX_RESOLUTION, "step": 8}),
                            "x": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),
                            "y": ("INT", {"default": 0, "min": 0, "max": MAX_RESOLUTION, "step": 8}),}}
                            
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/ScheduleNodes"

    def animate(self, text, max_frames, current_frame, clip, conditioning_to, gligen_textbox_model, width, height, x, y):
        inputText = str("{"+text+"}") #format the input so it's valid json
        animation_prompts = json.loads(inputText.strip())
        current_prompt, next_prompt, current_weight = self.interpolate_prompts(animation_prompts, max_frames, int(current_frame), clip)
        if np.isnan(current_weight):  # Check if current_weight is NaN
            print("single")
            return (self.append(list(conditioning_to), clip, gligen_textbox_model, str(current_prompt), width, height, x, y),)
        else:
            print("multiple")
            return (self.appendMultiple(list(conditioning_to), clip, gligen_textbox_model, str(current_prompt), str(next_prompt), width, height, x, y, current_weight),)



    def append(self, conditioning_to, clip, gligen_textbox_model, text, width, height, x, y):
        c = []
        cond, cond_pooled = clip.encode_from_tokens(clip.tokenize(text), return_pooled=True)
        print("append: \n", text)
        for t in conditioning_to:
            n = [t[0], t[1].copy()]
            position_params = [(cond_pooled, height // 8, width // 8, y // 8, x // 8)]
            prev = []
            if "gligen" in n[1]:
                prev = n[1]['gligen'][2]

            n[1]['gligen'] = ("position", gligen_textbox_model, prev + position_params)
            c.append(n)
        print("COND: \n",c)
        #cond, gligen = c[0][0], c[0][1]['gligen'] # separate the tensor_to
        #cond_gligen = (cond, {'gligen': gligen}) # combine the averaged result and gligen
        return (c, )

    def appendMultiple(self, conditioning_to, clip, gligen_textbox_model, text_to, text_from, width, height, x, y, weight): # returns a CONDITIONING value from addWeighted
        c = []
        print("append multiple: \n",text_to,"\n", text_from)
        cond_to, cond_pooled_to = clip.encode_from_tokens(clip.tokenize(text_to), return_pooled=True)
        cond_from, cond_pooled_from = clip.encode_from_tokens(clip.tokenize(text_from), return_pooled=True)
        cond_avr = (addWeighted(list(cond_pooled_to), list(cond_pooled_from), weight),)
        for t in conditioning_to:
            n = [t[0], t[1].copy()]
            position_params = [(cond_avr, height // 8, width // 8, y // 8, x // 8)]
            prev = []
            if "gligen" in n[1]:
                prev = n[1]['gligen'][2]

            n[1]['gligen'] = ("position", gligen_textbox_model, prev + position_params)
            c.append(n)

        #cond_to, gligen = c[0][0], c[0][1]['gligen'] # separate the tensor_to
        #cond_from = ([[clip.encode(str(text2)), {}]], ) # encode second prompt
        # # perform compostable diffusion
        #cond_gligen = (cond_avr, {'gligen': gligen}) # combine the averaged result and gligen

        return (c,)
        
    def interpolate_prompts(self, animation_prompts, max_frames, current_frame, clip): #parse the conditioning strength and determine in-betweens.
        #Get prompts sorted by keyframe
        max_f = max_frames #needed for numexpr even though it doesn't look like it's in use.
        parsed_animation_prompts = {}
        for key, value in animation_prompts.items():
            if check_is_number(key):  #default case 0:(1 + t %5), 30:(5-t%2)
                parsed_animation_prompts[key] = value
            else:  #math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                parsed_animation_prompts[int(numexpr.evaluate(key))] = value
        
        sorted_prompts = sorted(parsed_animation_prompts.items(), key=lambda item: int(item[0]))
        
        #Setup containers for interpolated prompts
        cur_prompt_series = pd.Series([np.nan for a in range(max_frames)])
        nxt_prompt_series = pd.Series([np.nan for a in range(max_frames)])

        #simple array for strength values
        weight_series = [np.nan] * max_frames

        #in case there is only one keyed promt, set all prompts to that prompt
        if len(sorted_prompts) - 1 == 0:
            for i in range(0, len(cur_prompt_series)-1):           
                current_prompt = sorted_prompts[0][1]           
                cur_prompt_series[i] = current_prompt
                nxt_prompt_series[i] = current_prompt

        # For every keyframe prompt except the last
        for i in range(0, len(sorted_prompts) - 1):
            # Get current and next keyframe
            current_key = int(sorted_prompts[i][0])
            next_key = int(sorted_prompts[i + 1][0])


            # Ensure there's no weird ordering issues or duplication in the animation prompts
            # (unlikely because we sort above, and the json parser will strip dupes)
            if current_key >= next_key:
                print(f"WARNING: Sequential prompt keyframes {i}:{current_key} and {i + 1}:{next_key} are not monotonously increasing; skipping interpolation.")
                continue
    
            # Get current and next keyframes' positive and negative prompts (if any)
            current_prompt = sorted_prompts[i][1]
            next_prompt = sorted_prompts[i + 1][1]
            
            # Calculate how much to shift the weight from current to next prompt at each frame.
            weight_step = 1 / (next_key - current_key)
    
            for f in range(current_key, next_key):
                next_weight = weight_step * (f - current_key)
                current_weight = 1 - next_weight
                
                #add the appropriate prompts and weights to their respective containers.
                cur_prompt_series[f] = ''
                nxt_prompt_series[f] = ''
                weight_series[f] = 0.0

                cur_prompt_series[f] += current_prompt
                nxt_prompt_series[f] += next_prompt

                weight_series[f] += current_weight
        
        #Evaluate the current and next prompt's expressions
        cur_prompt_series[current_frame] = prepare_prompt(cur_prompt_series[current_frame], max_frames, current_frame)
        nxt_prompt_series[current_frame] = prepare_prompt(nxt_prompt_series[current_frame], max_frames, current_frame)       

        #Show the to/from prompts with evaluated expressions for transparency.
        print(" current prompt: ",cur_prompt_series[current_frame], "\n","next prompt: ", nxt_prompt_series[current_frame], "\n", "strength : ", weight_series[current_frame])

        #Output methods depending if the prompts are the same or if the current frame is a keyframe.
        #if it is an in-between frame and the prompts differ, compostable diffusion will be performed.
        #this altered function returns text instead of conditioning so the gligen functions can operate.
        if str(cur_prompt_series[current_frame]) == str(nxt_prompt_series[current_frame]):
            return ((str(cur_prompt_series[current_frame])), np.nan, np.nan)
        if weight_series[current_frame] == 1:
            return ((str(cur_prompt_series[current_frame])), np.nan, np.nan) #Will probably never trigger but I'm paranoid
        if weight_series[current_frame] == 0:
            return ((str(nxt_prompt_series[current_frame])), np.nan, np.nan)
        else:
            return (str(cur_prompt_series[current_frame]), str(nxt_prompt_series[current_frame]), weight_series[current_frame])

#This node parses the user's test input into 
#interpolated floats. Expressions can be input 
#and evaluated.            
class ValueSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),
                             "current_frame": ("INT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0}),
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
