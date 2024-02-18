"""
Module to load PyTorch models based on architecture names, utilizing fuzzy matching to suggest correct model names.
"""

import torchvision.models.detection as detection_models
from fuzzywuzzy import process

def find_best_fuzzy_match(word: str, possibilities: list) -> tuple:
    """
    Finds the best match for a given word from a list of possibilities using fuzzy matching.

    Args:
        word (str): The word to match.
        possibilities (list): A list of possible matches.

    Returns:
        tuple: The best match and its score.

    Raises:
        ValueError: If no match is found.
    """
    best_match = process.extractOne(word, possibilities)
    if best_match:
        return best_match

    raise ValueError("No suitable match found.")

def load_model(architecture: str, pretrained: bool = False):
    """
    Loads a PyTorch model based on the given architecture name.

    Args:
        architecture (str): Name of the model architecture to load.
        pretrained (bool): Whether to load a pretrained model.

    Returns:
        A PyTorch model instance.

    Raises:
        ValueError: If the architecture name does not match any known models.
    """
    #Dict to map and load right architecture class
    architecture_map = {
    ##Object Detection Models##
    'fasterrcnn_resnet50_fpn_v2':detection_models.fasterrcnn_resnet50_fpn_v2,
    'fasterrcnn_resnet50_fpn':detection_models.fasterrcnn_resnet50_fpn,
    'fcos_resnet50_fpn':detection_models.fcos_resnet50_fpn,
    'fasterrcnn_mobilenet_v3_large_320_fpn':detection_models.fasterrcnn_mobilenet_v3_large_320_fpn,
    'fasterrcnn_mobilenet_v3_large_fpn':detection_models.fasterrcnn_mobilenet_v3_large_fpn,
    'retinanet_resnet50_fpn_v2':detection_models.retinanet_resnet50_fpn_v2,
    'retinanet_resnet50_fpn':detection_models.retinanet_resnet50_fpn,
    'ssd300_vgg16':detection_models.ssd300_vgg16,
    'ssdlite320_mobilenet_v3_large':detection_models.ssdlite320_mobilenet_v3_large,
    ##Instance Segmentation Models##
    'maskrcnn_resnet50_fpn_v2':detection_models.maskrcnn_resnet50_fpn_v2,
    'MaskRCNN_ResNet50_FPN_Weights':detection_models.MaskRCNN_ResNet50_FPN_Weights
    }

    if architecture in architecture_map:
        return architecture_map[architecture](pretrained=pretrained)

    possible_match = find_best_fuzzy_match(architecture, list(architecture_map.keys()))
    raise ValueError(f"Unsupported architecture: {architecture}. Did you mean: `{possible_match[0]}`?")
