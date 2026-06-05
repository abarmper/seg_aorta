"""SEG.A aorta lumen inference pipeline + post-hoc explainability."""
from sega.model import build_segresnet, load_checkpoint
from sega.transforms import make_preprocess, make_postprocess
from sega.inference import run_inference, run_mc_inference, prediction_to_sitk
from sega.explain import SegGradCAM, mc_dropout_uncertainty
from sega.mesh import create_meshes, vertices_array_to_physical
from sega import viz

__all__ = [
    "build_segresnet",
    "load_checkpoint",
    "make_preprocess",
    "make_postprocess",
    "run_inference",
    "run_mc_inference",
    "prediction_to_sitk",
    "SegGradCAM",
    "mc_dropout_uncertainty",
    "create_meshes",
    "vertices_array_to_physical",
    "viz",
]
