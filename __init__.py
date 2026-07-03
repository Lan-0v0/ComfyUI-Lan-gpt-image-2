"""
ComfyUI-Lan-gpt-image-2
=======================
A single-node ComfyUI plugin for advanced GPT image generation and editing
through any OpenAI-compatible API endpoint.

Key feature: configurable base_url — use official OpenAI, a local proxy
(http://localhost:8317/v1), or any compatible endpoint.

Install: copy this folder into ComfyUI/custom_nodes/ and restart ComfyUI.
"""

from .nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

# Web UI extension directory (tooltips, layout enhancements)
WEB_DIRECTORY = "./web"

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
