
# FizzNodes
Scheduled prompts, scheduled float/int values and wave function nodes for animations and utility. compatable with https://www.framesync.xyz/ and https://www.chigozie.co.uk/keyframe-string-generator/ for audio synced animations in [Comfyui](https://github.com/comfyanonymous/ComfyUI).

**  Please see the [Fizznodes wiki](https://github.com/FizzleDorf/ComfyUI_FizzNodes/wiki) for instructions on usage of these nodes as well as handy resources you can use in your projects! **


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

## Examples
Some examples using the prompt and value schedulers using base comfyui.

### Simple Animation Workflow
This example showcases making animations with only scheduled prompts. This method only uses 4.7 GB of memory and makes use of deterministic samplers (Euler in this case). 


![output](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/82f21ab2-209c-43d7-a202-67d99fd3c823)


Drag and drop the image in this link into ComfyUI to load the workflow or save the image and load it using the load button.

[Txt2_Img_Example](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/8899f25e-fbc8-423c-bef2-e7c5a91fb7f4)


### Noisy Latent Comp Workflow
This example showcases the [Noisy Laten Composition](https://comfyanonymous.github.io/ComfyUI_examples/noisy_latent_composition/) workflow. The value schedule node schedules the latent composite node's x position. You can also animate the subject while the composite node is being schedules as well!

![output](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/6ffe1078-1869-4b7a-990f-902b7eafd67d)


Drag and drop the image in this link into ComfyUI to load the workflow or save the image and load it using the load button.

[Latent_Comp_Example](https://github.com/FizzleDorf/ComfyUI_FizzNodes/assets/46942135/410fbd99-d06e-489a-b6f5-3b747acd3740)


## Helpful tools

Just a list of tools that you may find handy using these nodes.

Link | Description
--- | --- 
[Desmos Graphing Calculator](https://www.desmos.com/calculator) | online graphing calculator. Handy for visualizing expressions.
[Keyframe String Generator](https://www.chigozie.co.uk/keyframe-string-generator/) | custom keyframe string generator that is compatable with the valueSchedule node.
[Audio framesync](https://www.framesync.xyz/) | Audio sync wave functions. Exports keyframes for the valueSchedule node.
[SD-Parseq](https://github.com/rewbs/sd-parseq) | A powerful scheduling tool for audio sync and easy curve manupulation (my personal fave!)
-----

## Acknowledgments

**A special thanks to:**

-The developers of [Deforum](https://github.com/deforum-art/sd-webui-deforum) for providing code for these nodes and being overall awesome people!

-Comfyanonamous and the rest of the [ComfyUI](https://github.com/comfyanonymous/ComfyUI/tree/master) contributors for a fantastic UI!

-All the friends I met along the way that motivate me into action!

-and you the user! I hope you have fun using these nodes and exploring latent space.
