"""ComfyUI loads this folder as a package. Nothing here uses the network."""

WEB_DIRECTORY = "./web"

# pytest imports this file as a module without a package, where the relative
# imports cannot work. ComfyUI always imports it as a package.
if __package__:
    from .architect_inference import routes
    from .architect_inference.config import comfy_config
    from .architect_inference.nodes import NODE_CLASS_MAPPINGS, NODE_DISPLAY_NAME_MAPPINGS

    routes.register(comfy_config)

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS", "WEB_DIRECTORY"]
