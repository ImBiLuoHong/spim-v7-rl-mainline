"""
Deprecated stub for the retired training-engine model builder.

Official training uses:
    src.modeling.builders.model_builder.ModelBuilder

Legacy implementation archived at:
    src/training/legacy/model_builder_legacy.py
"""


def build_model(*args, **kwargs):
    raise RuntimeError(
        "src.training.engine.model_builder.build_model is deprecated. "
        "Official training must build models through src.modeling.builders.model_builder.ModelBuilder."
    )
