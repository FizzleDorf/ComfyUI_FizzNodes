# Made by Davemane42#0042 for ComfyUI
import os
import subprocess
import importlib.util
import sys
import filecmp
import shutil

import __main__

python = sys.executable


extentions_folder = os.path.join(os.path.dirname(os.path.realpath(__main__.__file__)),
                                 "web" + os.sep + "extensions" + os.sep + "FizzleDorf")
javascript_folder = os.path.join(os.path.dirname(os.path.realpath(__file__)), "javascript")

if not os.path.exists(extentions_folder):
    print('Making the "web\extensions\FizzleDorf" folder')
    os.makedirs(extentions_folder)

result = filecmp.dircmp(javascript_folder, extentions_folder)

if result.left_only or result.diff_files:
    print('Update to javascripts files detected')
    file_list = list(result.left_only)
    file_list.extend(x for x in result.diff_files if x not in file_list)

    for file in file_list:
        print(f'Copying {file} to extensions folder')
        src_file = os.path.join(javascript_folder, file)
        dst_file = os.path.join(extentions_folder, file)
        if os.path.exists(dst_file):
            os.remove(dst_file)
        #print("disabled")
        shutil.copy(src_file, dst_file)


def is_installed(package, package_overwrite=None):
    try:
        spec = importlib.util.find_spec(package)
    except ModuleNotFoundError:
        pass

    package = package_overwrite or package

    if spec is None:
        print(f"Installing {package}...")
        command = f'"{python}" -m pip install {package}'
  
        result = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.PIPE, shell=True, env=os.environ)

        if result.returncode != 0:
            print(f"Couldn't install\nCommand: {command}\nError code: {result.returncode}")

from .WaveNodes import Lerp, SinWave, InvSinWave, CosWave, InvCosWave, SquareWave, SawtoothWave, TriangleWave, AbsCosWave, AbsSinWave
from .ScheduledNodes import (
    ValueSchedule, PromptSchedule, PromptScheduleNodeFlow, PromptScheduleNodeFlowEnd, PromptScheduleEncodeSDXL,
    StringSchedule, BatchPromptSchedule, BatchValueSchedule, BatchPromptScheduleEncodeSDXL, BatchStringSchedule,
    BatchValueScheduleLatentInput, BatchPromptScheduleEncodeSDXLLatentInput, BatchPromptScheduleLatentInput,
    ImagesFromBatchSchedule,
    #, BatchPromptScheduleNodeFlowEnd #, BatchGLIGENSchedule
)
from .FrameNodes import FrameConcatenate, InitNodeFrame, NodeFrame, StringConcatenate
from .HelperNodes import ConcatStringSingle, convertKeyframeKeysToBatchKeys, CalculateFrameOffset

NODE_CLASS_MAPPINGS = {
    "Lerp": Lerp,
    "SinWave": SinWave,
    "InvSinWave": InvSinWave,
    "CosWave": CosWave,
    "InvCosWave": InvCosWave,
    "SquareWave":SquareWave,
    "SawtoothWave": SawtoothWave,
    "TriangleWave": TriangleWave,
    "AbsCosWave": AbsCosWave,
    "AbsSinWave": AbsSinWave,
    "PromptSchedule": PromptSchedule,
    "ValueSchedule": ValueSchedule,
    "PromptScheduleNodeFlow": PromptScheduleNodeFlow,
    "PromptScheduleNodeFlowEnd": PromptScheduleNodeFlowEnd,
    "PromptScheduleEncodeSDXL":PromptScheduleEncodeSDXL,
    "StringSchedule":StringSchedule,
    "BatchPromptSchedule": BatchPromptSchedule,
    "BatchValueSchedule": BatchValueSchedule,
    "BatchPromptScheduleEncodeSDXL": BatchPromptScheduleEncodeSDXL,
    "BatchStringSchedule": BatchStringSchedule,
    "BatchValueScheduleLatentInput": BatchValueScheduleLatentInput,
    "BatchPromptScheduleSDXLLatentInput":BatchPromptScheduleEncodeSDXLLatentInput,
    "BatchPromptScheduleLatentInput":BatchPromptScheduleLatentInput,
    "ImagesFromBatchSchedule":ImagesFromBatchSchedule,
    #"BatchPromptScheduleNodeFlowEnd":BatchPromptScheduleNodeFlowEnd,
    #"BatchGLIGENSchedule": BatchGLIGENSchedule,

    "StringConcatenate":StringConcatenate,
    "Init FizzFrame":InitNodeFrame,
    "FizzFrame":NodeFrame,
    "FizzFrameConcatenate":FrameConcatenate,

    "ConcatStringSingle": ConcatStringSingle,
    "convertKeyframeKeysToBatchKeys": convertKeyframeKeysToBatchKeys,
    "CalculateFrameOffset":CalculateFrameOffset,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lerp": "Lerp 📅🅕🅝",
    "SinWave": "SinWave 📅🅕🅝",
    "InvSinWave": "InvSinWave 📅🅕🅝",
    "CosWave": "CosWave 📅🅕🅝",
    "InvCosWave": "InvCosWave 📅🅕🅝",
    "SquareWave":"SquareWave 📅🅕🅝",
    "SawtoothWave": "SawtoothWave 📅🅕🅝",
    "TriangleWave": "TriangleWave 📅🅕🅝",
    "AbsCosWave": "AbsCosWave 📅🅕🅝",
    "AbsSinWave": "AbsSinWave 📅🅕🅝",
    "PromptSchedule": "Prompt Schedule 📅🅕🅝",
    "ValueSchedule": "Value Schedule 📅🅕🅝",
    "PromptScheduleNodeFlow": "Prompt Schedule NodeFlow 📅🅕🅝",
    "PromptScheduleNodeFlowEnd": "Prompt Schedule NodeFlow End 📅🅕🅝",
    "StringSchedule":"String Schedule 📅🅕🅝",
    "StringConcatenate":"String Concatenate 📅🅕🅝",
    "Init FizzFrame":"Init Node Frame 📅🅕🅝",
    "FizzFrame":"Node Frame 📅🅕🅝",
    "FizzFrameConcatenate":"Frame Concatenate 📅🅕🅝",
    "BatchPromptSchedule": "Batch Prompt Schedule 📅🅕🅝",
    "BatchValueSchedule": "Batch Value Schedule 📅🅕🅝",
    "PromptScheduleEncodeSDXL": "Prompt Schedule SDXL 📅🅕🅝",
    "BatchPromptScheduleEncodeSDXL": "Batch Prompt Schedule SDXL 📅🅕🅝",
    "BatchStringSchedule": "Batch String Schedule 📅🅕🅝",
    "BatchValueScheduleLatentInput": "Batch Value Schedule (Latent Input) 📅🅕🅝",
    "BatchPromptScheduleSDXLLatentInput": "Batch Prompt Schedule SDXL (Latent Input) 📅🅕🅝",
    "BatchPromptScheduleLatentInput": "Batch Prompt Schedule (Latent Input) 📅🅕🅝",
    "ImageBatchFromValueSchedule":"Image Batch From Value Schedule 📅🅕🅝",
    "ConcatStringSingle": "Concat String (Single) 📅🅕🅝",
    "convertKeyframeKeysToBatchKeys":"Keyframe Keys To Batch Keys 📅🅕🅝",
    "SelectFrameNumber":"Select Frame Number 📅🅕🅝",
    "CalculateFrameOffset":"Calculate Frame Offset 📅🅕🅝",
    "ImagesFromBatchSchedule":"Image Select Schedule 📅🅕🅝",
}
print('\033[34mFizzleDorf Custom Nodes: \033[92mLoaded\033[0m')
