# -*- coding: utf-8 -*-
# Copyright (c) 2025 Salvador E. Tropea
# Copyright (c) 2025 Instituto Nacional de Tecnologïa Industrial
# License: GPL-3.0
# Project: ComfyUI-AudioBatch
from .src.nodes import nodes_audio, main_logger
from seconohe.register_nodes import register_nodes
from seconohe import JS_PATH


NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS = register_nodes(main_logger, [nodes_audio])
WEB_DIRECTORY = JS_PATH
__all__ = ['NODE_CLASS_MAPPINGS', 'NODE_DISPLAY_NAME_MAPPINGS']
