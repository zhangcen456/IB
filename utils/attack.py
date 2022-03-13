from textattack.datasets import HuggingFaceDataset
from data.instance import InputInstance
from typing import List, Dict
import random
from textattack.attack_recipes import TextFoolerJin2019,HotFlipEbrahimi2017,DeepWordBugGao2018,TextBuggerLi2018
from bertattack import BERTAttackLi2020
from textattack.transformations import WordSwapEmbedding
from textattack.constraints.semantics import WordEmbeddingDistance
from textattack.constraints.overlap import MaxWordsPerturbed
from textattack.goal_functions import UntargetedClassification
from textattack.constraints.pre_transformation import (
    InputColumnModification,
    RepeatModification,
    StopwordModification,
)
from textattack.constraints.semantics.sentence_encoders import UniversalSentenceEncoder
from textattack import Attack
from textattack.models.wrappers import ModelWrapper,PyTorchModelWrapper
import torch

class CustomModelWrapper(PyTorchModelWrapper):
    def __init__(self,model,tokenizer):
        super(CustomModelWrapper,self).__init__(model,tokenizer)

    def __call__(self,text_input_list):
        inputs_dict = self.tokenizer(
            text_input_list,
            truncation=True,
            padding=True,
            return_tensors="pt",
        )
        model_device = next(self.model.parameters()).device
        inputs_dict.to(model_device)

        with torch.no_grad():
            outputs = self.model(**inputs_dict)

        if isinstance(outputs,tuple):
            return outputs[-1]#model-h,model-bh

        if isinstance(outputs,torch.Tensor):
            return outputs#baseline

        if isinstance(outputs[0], str):
            # HuggingFace sequence-to-sequence models return a list of
            # string predictions as output. In this case, return the full
            # list of outputs.
            return outputs
        else:
            # HuggingFace classification models return a tuple as output
            # where the first item in the tuple corresponds to the list of
            # scores for each input.
            return outputs.logits

def build_attacker(model,args):
    if (args['attack_method'] == 'textfooler'):
        attacker=TextFoolerJin2019.build(model)
    elif (args['attack_method'] == 'textbugger'):
        attacker=TextBuggerLi2018.build(model)
    elif (args['attack_method'] == 'bertattack'):
        attacker=BERTAttackLi2020.build(model)
    else:
        attacker=TextFoolerJin2019.build(model)
    if(args['modify_ratio']!=0):
        attacker.constraints.append(MaxWordsPerturbed(max_percent=args['modify_ratio']))
    return attacker