#These nodes were made using code from the Deforum extension for A1111 webui
#You can find the project here: https://github.com/deforum-art/sd-webui-deforum
from cmath import nan
from inspect import currentframe
import numexpr
import torch
import numpy as np
import pandas as pd
import re
import simplejson as json

# used by both nodes
def check_is_number(value):
    float_pattern = r'^(?=.)([+-]?([0-9]*)(\.([0-9]+))?)$'
    return re.match(float_pattern, value)

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
                            "current_frame": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0}),}}
                            
    RETURN_TYPES = ("CONDITIONING",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/ScheduleNodes"

    def animate(self, text, max_frames, current_frame, clip,):
        #format the input so it's valid json
        inputText = str("{"+text+"}")
        animation_prompts = json.loads(inputText.strip())
        return self.interpolate_prompts(animation_prompts, max_frames, int(current_frame), clip) #return a conditioning value   
    

    def addWeighted(self, conditioning_to, conditioning_from, conditioning_to_strength):
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

    #parse the conditioning strength and determine in-betweens.
    def parse_weight(self, match, frame=0, max_frames=0) -> float:
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
            print(float(numexpr.evaluate(w_raw[1:-1])))
            return float(numexpr.evaluate(w_raw[1:-1]))
    
        
    def prepare_prompt(self, prompt_series, max_frames, frame_idx):
        max_f = max_frames - 1
        pattern = r'`.*?`' #set so the expression will be read between two backticks (``)
        regex = re.compile(pattern)
        prompt_parsed = str(prompt_series)
        for match in regex.finditer(prompt_parsed):
            matched_string = match.group(0)
            parsed_string = matched_string.replace('t', f'{frame_idx}').replace("max_f", f"{max_f}").replace('`', '') #replace t, max_f and `` respectively
            parsed_value = numexpr.evaluate(parsed_string)
            prompt_parsed = prompt_parsed.replace(matched_string, str(parsed_value))
        print(prompt_parsed.strip())
        return prompt_parsed.strip()
    
    def interpolate_prompts(self, animation_prompts, max_frames, current_frame, clip):
        #Get prompts sorted by keyframe
        print("animation_prompts :", animation_prompts)
        max_f = max_frames #needed for numexpr even though it doesn't look like it's in use.
        parsed_animation_prompts = {}
        for key, value in animation_prompts.items():
            if check_is_number(key):  #default case 0:(1 + t %5), 30:(5-t%2)
                parsed_animation_prompts[key] = value
            else:  #math on the left hand side case 0:(1 + t %5), maxKeyframes/2:(5-t%2)
                parsed_animation_prompts[int(numexpr.evaluate(key))] = value
        
        print("parsed_animation_prompts :", parsed_animation_prompts)
        sorted_prompts = sorted(parsed_animation_prompts.items(), key=lambda item: int(item[0]))
        
        #Setup containers for interpolated prompts
        cur_prompt_series = pd.Series([np.nan for a in range(max_frames)])
        nxt_prompt_series = pd.Series([np.nan for a in range(max_frames)])

        #simple array for strength values
        weight_series = [np.nan] * max_frames

        print("sorted_prompts:", sorted_prompts)

        #in case there is only one keyed promt, set all prompts to that prompt
        if len(sorted_prompts) - 1 == 0:
            for i in range(0, len(cur_prompt_series)-1):           
                current_prompt = sorted_prompts[0][1]           
                cur_prompt_series[i] = current_prompt
                nxt_prompt_series[i] = current_prompt

        # For every keyframe prompt except the last
        for i in range(0, len(sorted_prompts) - 1):
            print("i :", i)
            # Get current and next keyframe
            current_key = int(sorted_prompts[i][0])
            print("curKey:",current_key)
            next_key = int(sorted_prompts[i + 1][0])
            print("nxtKey:", next_key)
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
                
                print(current_key)

                #add the appropriate prompts and weights to their respective containers.
                cur_prompt_series[f] = ''
                nxt_prompt_series[f] = ''
                weight_series[f] = 0.0
                
                cur_prompt_series[f] += current_prompt
                nxt_prompt_series[f] += next_prompt
                weight_series[f] += current_weight  

                #make sure the prompt is set to what it was last if there isn't any in the series.
                if cur_prompt_series[f] == nan:
                    cur_prompt_series[f] = cur_prompt_series[f-1]
                    print("cur_prompt_series[f]:",cur_prompt_series[f])
                if nxt_prompt_series[f] == nan:
                    nxt_prompt_series[f] = nxt_prompt_series[f-1]

            
        
        #Evaluate the current and next prompt's expressions
        cur_prompt_series[current_frame] = self.prepare_prompt(cur_prompt_series[current_frame], max_frames, current_frame)
        nxt_prompt_series[current_frame] = self.prepare_prompt(nxt_prompt_series[current_frame], max_frames, current_frame)       

        print("cur_prompt_series[current_frame]: ",cur_prompt_series[current_frame])
        print("nxt_prompt_series[current_frame]: ",nxt_prompt_series[current_frame])

        #Output methods depending if the prompts are the same or if the current frame is a keyframe.
        #if it is an in-between frame and the prompts differ, compostable diffusion will be performed.
        if str(cur_prompt_series[current_frame]) == str(nxt_prompt_series[current_frame]):
            #Debug
            print("both equal, only curr")
            return ([[clip.encode(str(cur_prompt_series[current_frame])), {}]], )
        if weight_series[current_frame] == 1:
            #Debug
            print("only curr")
            return ([[clip.encode(str(cur_prompt_series[current_frame])), {}]], )
        if weight_series[current_frame] == 0:
            #Debug
            print("only next")
            return ([[clip.encode(str(nxt_prompt_series[current_frame])), {}]], )
        else:
            #Debug
            print("both")
            return self.addWeighted(list([[clip.encode(str(cur_prompt_series[current_frame])), {}]], ), list([[clip.encode(str(nxt_prompt_series[current_frame])), {}]], ), weight_series[current_frame])


#This node parses the user's test input into interpolated floats. 
#Expressions can be input and evaluated.            
class ValueSchedule:
    @classmethod
    def INPUT_TYPES(s):
        return {"required": {"text": ("STRING", {"multiline": True}),
                             "max_frames": ("INT", {"default": 120.0, "min": 1.0, "max": 9999.0, "step": 1.0}),
                             "current_frame": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 9999.0, "step": 1.0}),}}
    RETURN_TYPES = ("FLOAT",)
    FUNCTION = "animate"

    CATEGORY = "FizzNodes/ScheduleNodes"
    
    def animate(self, text, max_frames, current_frame,):
        t = self.get_inbetweens(self.parse_key_frames(text, max_frames), max_frames)
        cFrame = int(current_frame) #current_frame is currently recieved as a float hence the cast to int
        return (t[cFrame],)
    
    def sanitize_value(self, value):
        return value.replace("'","").replace('"',"").replace('(',"").replace(')',"")

    def get_inbetweens(self, key_frames, max_frames, integer=False, interp_method='Linear', is_single_string = False):
        key_frame_series = pd.Series([np.nan for a in range(max_frames)])
        max_f = max_frames -1
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
            max_f = max_frames -1
            frame = int(self.sanitize_value(frameParam[0])) if check_is_number(self.sanitize_value(frameParam[0].strip())) else int(numexpr.evaluate(frameParam[0].strip().replace("'","",1).replace('"',"",1)[::-1].replace("'","",1).replace('"',"",1)[::-1]))
            frames[frame] = frameParam[1].strip()
        if frames == {} and len(string) != 0:
            raise RuntimeError('Key Frame string not correctly formatted')
        return frames