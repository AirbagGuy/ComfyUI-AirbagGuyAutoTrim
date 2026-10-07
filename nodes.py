import numpy as np
import torch

from .trim_core import ALPHA_HANDLING, MASK_CONVENTIONS, MODES, trim_batch


class AirbagGuyAutoTrim:
    """
    Trims the empty border around a subject: a transparent background (alpha),
    a flat background color, or an explicit mask. Adds padding and outputs the
    trimmed image, the subject mask, and the final width and height.
    """

    @classmethod
    def INPUT_TYPES(cls):
        return {
            "required": {
                "image": ("IMAGE",),
                "mode": (MODES, {"default": "Both",
                                 "tooltip": "Which edges to trim."}),
                "padding": ("INT", {"default": 32, "min": 0, "max": 4096, "step": 1,
                                    "tooltip": "Pixels added back around the subject on trimmed edges."}),
                "tolerance": ("FLOAT", {"default": 0.06, "min": 0.0, "max": 1.0, "step": 0.005,
                                        "tooltip": "How different from the background color a pixel must be to count as subject (color mode only)."}),
                "min_coverage": ("FLOAT", {"default": 0.003, "min": 0.0, "max": 0.5, "step": 0.001,
                                           "tooltip": "Share of a row/column that must be subject before it counts. Ignores stray specks."}),
                "shared_crop_for_batch": ("BOOLEAN", {"default": True,
                                                      "tooltip": "Use one crop box for the whole batch so frames stay aligned."}),
                "alpha_handling": (ALPHA_HANDLING, {"default": "Use alpha channel",
                                                    "tooltip": "Use alpha to find the subject when the image has one, ignore it, or require it."}),
                "alpha_threshold": ("FLOAT", {"default": 0.5, "min": 0.0, "max": 1.0, "step": 0.01,
                                              "tooltip": "Alpha above this counts as subject."}),
                "flatten_background": ("BOOLEAN", {"default": False,
                                                   "tooltip": "Composite a transparent image over background_color and output RGB."}),
                "alpha_floor": ("FLOAT", {"default": 0.0, "min": 0.0, "max": 1.0, "step": 0.01,
                                          "tooltip": "Alpha below this is set to fully transparent (cleans faint haze)."}),
                "background_color": ("STRING", {"default": "#FFFFFF",
                                                "tooltip": "Background color as hex, or 'auto' to sample the image border. Also used for padding and flattening."}),
                "mask_convention": (MASK_CONVENTIONS, {"default": MASK_CONVENTIONS[0],
                                                       "tooltip": "How to read the optional mask input. Load Image masks are white where transparent."}),
            },
            "optional": {
                "mask": ("MASK",),
            },
        }

    RETURN_TYPES = ("IMAGE", "MASK", "INT", "INT")
    RETURN_NAMES = ("image", "mask", "width", "height")
    FUNCTION = "trim"
    CATEGORY = "AirbagGuy/image"

    def trim(self, image, mode, padding, tolerance, min_coverage, shared_crop_for_batch,
             alpha_handling, alpha_threshold, flatten_background, alpha_floor,
             background_color, mask_convention, mask=None):
        images = [img for img in image.detach().cpu().float().numpy()]

        masks = None
        if mask is not None:
            m = mask.detach().cpu().float().numpy()
            if m.ndim == 2:
                m = m[None]
            if m.shape[0] == 1 and len(images) > 1:
                m = np.repeat(m, len(images), axis=0)
            if m.shape[0] != len(images):
                raise ValueError(f"Mask batch ({m.shape[0]}) does not match image batch ({len(images)}).")
            masks = [x for x in m]

        out_images, out_masks = trim_batch(
            images, masks,
            mode=mode, padding=padding, tolerance=tolerance, min_coverage=min_coverage,
            shared_crop_for_batch=shared_crop_for_batch, alpha_handling=alpha_handling,
            alpha_threshold=alpha_threshold, flatten_background=flatten_background,
            alpha_floor=alpha_floor, background_color=background_color,
            mask_convention=mask_convention,
        )

        image_out = torch.from_numpy(np.stack(out_images).astype(np.float32))
        mask_out = torch.from_numpy(np.stack(out_masks).astype(np.float32))
        height, width = image_out.shape[1], image_out.shape[2]
        return (image_out, mask_out, int(width), int(height))


NODE_CLASS_MAPPINGS = {
    "AirbagGuyAutoTrim": AirbagGuyAutoTrim,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "AirbagGuyAutoTrim": "AirbagGuy Auto Trim",
}
