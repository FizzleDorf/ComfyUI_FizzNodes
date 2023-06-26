
# FizzNodes
Scheduled prompts, scheduled float/int values and wave function nodes for animations and utility. compatable with https://www.framesync.xyz/ and https://www.chigozie.co.uk/keyframe-string-generator/ for audio synced animations in [Comfyui](https://github.com/comfyanonymous/ComfyUI).

Example: *still in the oven*

-----

### Table of Contents  
- 1.0 [Installation](https://github.com/FizzleDorf/ComfyUI_FizzNodes#installation)

- 2.0 [Important Notes on Operation](https://github.com/FizzleDorf/ComfyUI_FizzNodes#important)

- 3.0 [Scheduled Nodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes#schedule-nodes)

  - 3.1 [Variables and Expressions](https://github.com/FizzleDorf/ComfyUI_FizzNodes#variables-and-expressions)

  - 3.2 [Promptschedule](https://github.com/FizzleDorf/ComfyUI_FizzNodes#promptschedule)

  - 3.3 [ValueSchedule](https://github.com/FizzleDorf/ComfyUI_FizzNodes#valuechedule)

- 4.0 [Wave Nodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes#wavenodes)

- 5.0 [Helpful Tools](https://github.com/FizzleDorf/ComfyUI_FizzNodes#helpful-tools)

- 6.0 [Acknowledgements](https://github.com/FizzleDorf/ComfyUI_FizzNodes#acknowledgments)
  
-----

## Installation

For the easiest install experience, install the [Comfyui Manager](https://github.com/ltdrdata/ComfyUI-Manager) and use that to automate the installation process.
Otherwise, to manually install, simply clone the repo into the custom_nodes directory with this command:
```
git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes.git
```
and install the requirements using:
```
.\python_embed\python.exe -s -m pip install -r requirements.txt
```
If you are using a venv, make sure you have it activated before installation and use:
```
pip install -r requirements.txt
```

Example | Instructions
---|---
![Fizznodes menu](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/e07fedba-648c-4300-a6ac-61873b1501ab)|The nodes will can be accessed in the FizzNodes section of the node menu. You can also use the node search to find the nodes you are looking for. 

-----

TODO:
- [x] fix runoff past last keyframe
- [x] add prepend/append inputs for prompt schedule
- [x] prompt weight variables
- [x] create readme
- [ ] Node flow method (there is an implementation although a bit annoying to convert inputs. I'll check this once that's sorted)
- [ ] workflow examples
- [ ] video examples
- [ ] Gligen support
- [ ] growable array of prompt weights
- [ ] attempt simplified scheduler node (another alternative)
- [ ] add more to this list

-----

## Important!

Instructions | Example
---|---
All of these nodes require the primitive nodes incremental output in the current_frame input. To set this up, simply right click on the node and convert current_frame to an input. Then, double click the input to add a primitive node. Set the node value control to increment and the value to 0. The primitive should look like this: | ![primitiveNode](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/c78063c8-8542-484b-a15c-1f47a8c2d489) 
The text inputs ```pre_text``` and ```app_text``` are for appending or prepending text to every scheduled prompt. The primitive that is made from double clicking the input is single line and might be a little inconvenient. I reccomend using the TextBox from these [modded nodes](https://github.com/Derfuu/Derfuu_ComfyUI_ModdedNodes) as the input for either of these inputs. This node suite also has a lot of math operator nodes that can come in handy when using these nodes. | ![text box](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/df865a30-cf1c-4479-9ca4-c319486aabc5)
The Prompt Scheduler has multiple options that need to be converted to an input in order to properly use them. The Prompt weight channels (```pw_a```, ```pw_b```, etc.) can take in the result from a Value scheduler giving full control of the token weight over time. | ![value example](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/24dbd807-920c-40ea-bb9c-8b8ce556402d)
An example setup that includes prepended text and two prompt weight variables would look something like this: | ![example setup](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/24156699-dfce-487b-a1a7-009d6c7a8507)

"Note: I used [keyframe string generator](https://www.chigozie.co.uk/keyframe-string-generator/) to manually set the animation curves in the value schedule."

-----

## Schedule Nodes

Example | Description
----- | -----
![ScheduleNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/36c4ff23-7bd1-48e2-9fb9-2549e9764535) | These nodes are a convenient way to schedule values over time. This includes anything with a float input, int input, prompts and prompt weights. This is done by interpolating the values at each keyframe and evaluating the expressions used in the inputs over time.


Both nodes contain a max_frames value that determines the size of the series. This only needs to be equal to or higher than your last keyed prompt/value.

-----

### Variables and expressions

For expressions, you can check out [supported numexpr operators and expressions](https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-operators) and use them in your prompt.

Both nodes use the same variables:
Variable | Definition 
--- | --- 
```t``` | current frame
```max_f``` | max frames

Prompt Schedule Only Variables include:
Variable | Definition 
--- | --- 
```pw_a``` | prompt weight A
```pw_b``` | prompt weight B
```pw_c``` | prompt weight C
```pw_d``` | prompt weight D

The value of these prompt weight variables depends on what you give as an input.

-----

### PromptSchedule
This node interpolates prompts and automates prompt weights over time using expressions in the prompt.

To keyframe a prompt, you need to format it correctly.

```
"#":"(prompt:`exp`)"
```

where ```#``` is the keyframe (as a whole number), ```prompt``` is your prompt, and ```exp``` is your expression.

The keyframe number needs to be enclosed by quotations (```""```) followed by a colon (```:```).
Your prompt also needs to be enclosed in quotations (```""```).
If you plan on having another keyframed prompt after this one, you need to place a comma (```,```) after the closing quote of your last prompt. If you don't do this you will get an error. If it is your last prompt do not place a comma as this will result in an error as well.

**Expressions in the prompt schedule must be enclosed using back ticks: ``` `` ``` not apostrpophes: ```''``` !!! If you are using prompt weight variables such as ```pw_a```, make sure it's enclosed inside backticks as well.**

An example of syntax is as follows:
```
"0": "1girl, solo, long grey hair, grey eyes, black sweater, (smiling:`(0.5+0.5*sin(t/12))`)",
"24": "1girl, solo, long grey hair, grey eyes, black sweater, (dancing:`pw_a`)",
"48": "1girl, solo, long grey hair, grey eyes, black sweater, (dancing:`pw_a`)",
"72": "1girl, solo, long grey hair, grey eyes, black sweater, (smiling:`(0.5+0.5*sin(t/max_f))`)"
```

To alleviate having to write the full prompt to every keyed frame, the prompts that stay the same through the whole animation can be prepended or appended to every prompt in the schedule using ```pre_text``` and ```app_text``` respectively. I would suggest using the text box suggested in the [important notes](https://github.com/FizzleDorf/ComfyUI_FizzNodes#important) section. Converting the above example would look like this:

pre_text  
```
1girl, solo, long grey hair, grey eyes, black sweater,  
```
Scheduled Text
```
"0": "(smiling:`(0.5+0.5*sin(t/12))`)",
"24": "(dancing:`pw_a`)",
"48": "(dancing:`pw_a`)",
"72": "(smiling:`(0.5+0.5*sin(t/max_f))`)"```
```

This will be the same output prompts as the first example provided, makes the prompt schedule easy to read and it's easy to edit.

-----

### ValueSchedule
This node interpolates float values as well as calculates expressions given by the user through the text input.

To keyframe a value, you need to format it correctly.

```
#: (value)
```

where ```#``` is you keyframe (as a whole number) and ```value``` is your value or expression.

A colon (```:```) needs to be placed between the key number and the value amd the value needs to be enclosed in parenthesis (```()```). If you plan on having a value after, make sure you have a comma (```,```) at the end of the keyed value or there will be an error.

An example of syntax is as follows:
```0: (0.0), 24: (0.8), 48: (6%t), 72: (-cos(0.5*t/12))```

-----

## WaveNodes
Example | Description
----- | -----
![WaveNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/21fd2e2d-af8d-4f8b-8b04-9175e4f00dce) | These nodes are simply wave functions that use the current frame for calculating the output. Less powerful than the schedule nodes but easy to use for beginners or for quick automation. The sawtooth wave (modulus) for example is a good way to set the same seed sequence for grids without using multiple ksamplers.

-----

## Helpful tools

Just a list of tools that you may find handy using these nodes.

Link | Description
--- | --- 
[Desmos Graphing Calculator](https://www.desmos.com/calculator) | online graphing calculator. Handy for visualizing expressions.
[Keyframe String Generator](https://www.chigozie.co.uk/keyframe-string-generator/) | custom keyframe string generator that is compatable with the valueSchedule node.
[Audio framesync](https://www.framesync.xyz/) | Audi sync wave functions. Exports keyframes for the valueSchedule node.

-----
## Acknowledgments

**A special thanks to:**

-The developers of [Deforum](https://github.com/deforum-art/sd-webui-deforum) for providing code for these nodes and being overall awesome people!

-Comfyanonamous and the rest of the [ComfyUI](https://github.com/comfyanonymous/ComfyUI/tree/master) contributors for a fantastic UI!

-All the friends I met along the way that motivate me into action!

-and you the user! I hope you have fun using these nodes and exploring latent space.
