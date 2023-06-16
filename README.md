
# FizzNodes
Custom animation and utility nodes for [Comfyui](https://github.com/comfyanonymous/ComfyUI)

Example: *still in the oven*


-----

## Installation
simply clone the repo into the custom_nodes directory with this command:
```
git clone https://github.com/FizzleDorf/ComfyUI_FizzNodes.git
```
The nodes will can be accessed in the FizzNodes subsection of the node menu

![Fizznodes menu](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/e07fedba-648c-4300-a6ac-61873b1501ab)

-----

## Important!

All of these nodes require the primitive nodes incremental output in the current_frame input. To set this up, simply right click on the node and convert current_frame to an input. Then, double click the input to add a primitive node. Set the node value control to increment and the value to 0. The primitive should look like this:

![primitiveNode](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/b55d041b-d5d1-487a-8994-c2ca95baf5f1)

-----

## Schedule Nodes

![ScheduleNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/36c4ff23-7bd1-48e2-9fb9-2549e9764535)


These nodes are a convenient way to schedule values over time. This includes anything with a float input, prompts and prompt weights. This is done by interpolating the values at each keyframe and evaluating the expressions used in the inputs.

Both nodes contail a max_frames value that determines the size of the series. This only needs to be higher than your last keyed prompt/value. Currently, if your current frame goes past your last keyed prompt/value, it will return nan. I'll be looking into a solution for this in time so for now make sure there is a keyed prompt/value on your last frame.

### Variables and expressions

For expressions, you can see what is available to use here: [supported numexpr operators and expressions](https://numexpr.readthedocs.io/en/latest/user_guide.html#supported-operators)

Both nodes use the same variables:
Variable | Definition 
--- | --- 
t | current frame
max_f | max frames

-----

### PromptSchedule
This node interpolates prompts and automates prompt weights over time using expressions in the prompt.

To keyframe a prompt, you need to format it correctly.

```
"#":"(prompt:`expression`)"
```

where ```#``` is the keyframe (as a whole number), ```prompt``` is your prompt, and ```expression``` is your expression.

The keyframe number needs to be enclosed by quotations (```""```) followed by a colon (```:```).
Your prompt also needs to be enclosed in quotations (```""```).
If you plan on having another keyframed prompt after this one, you need to place a comma (```,```) after the closing quote of your last prompt. If you don't do this you will get an error. If it is your last prompt do not place a comma as this will result in an error as well.

**Expressions in the prompt schedule must be enclosed using back ticks: ``` `` ``` not apostrpophes: ```''``` !!!**

An example of syntax is as follows:
```
"0": "1girl, solo, long grey hair, grey eyes, black sweater, (smiling:`(0.5+0.5*sin(t/12))`)",
"24": "1girl, solo, long grey hair, grey eyes, black sweater, (dancing:`(0.5+0.5*sin(t/12))`)",
"48": "1girl, solo, long grey hair, grey eyes, black sweater, (dancing:`(0.5+0.5*sin(t/12))`)"
"72": "1girl, solo, long grey hair, grey eyes, black sweater, (smiling:`(0.5+0.5*sin(t/max_f))`)",
```

-----

###ValueSchedule
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

![WaveNodes](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/21fd2e2d-af8d-4f8b-8b04-9175e4f00dce)

These nodes are simply wave functions that use the current frame for calculating the output. Less powerful than the schedule nodes but easy to use for beginners or for quick automation. The sawtooth wave (modulus) for example is a good way to set the same seed sequence for grids without using multiple ksamplers.

-----

## Helpful tools

Just a list of tools that you may find handy using these nodes.

Link | Description
--- | --- 
[Desmos Graphing Calculator](https://www.desmos.com/calculator) | online graphing calculator. Handy for visualizing expressions.
[Keyframe String Generator](https://www.chigozie.co.uk/keyframe-string-generator/) | custom keyframe string generator that is compatable with the valueSchedule node.
[Audio framesync](https://www.framesync.xyz/) | Audi sync wave functions. Exports keyframes for the valueSchedule node.

-----

TODO:
- [ ] fix runoff past last keyframe
- [ ] add prepend/append inputs for prompt schedule
- [ ] prompt weight variables
- [ ] edit readme (including examples)
- [ ] add more to this list
