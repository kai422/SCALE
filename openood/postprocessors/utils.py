from openood.utils import Config

from .scale_postprocessor import ScalePostprocessor

def get_postprocessor(config: Config):
    postprocessors = {
        'scale': ScalePostprocessor,
    }

    return postprocessors[config.postprocessor.name](config)
