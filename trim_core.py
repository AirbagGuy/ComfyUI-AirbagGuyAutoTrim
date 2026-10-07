"""
Pure-numpy trimming logic for AirbagGuy Auto Trim.

Kept separate from the ComfyUI node so it can be tested without torch.
Images are float32 arrays shaped (H, W, C) with values 0..1, where C is 3 (RGB)
or 4 (RGBA). Masks are float32 arrays shaped (H, W), where 1 = subject.
"""

import numpy as np

MODES = ["Both", "Horizontal only", "Vertical only"]
ALPHA_HANDLING = ["Use alpha channel", "Ignore alpha", "Alpha only"]
MASK_CONVENTIONS = [
    "Mask white = background (Load Image)",
    "Mask white = subject",
]


def parse_hex_color(value):
    """'#RRGGBB' / 'RRGGBB' / '#RGB' -> np.array([r, g, b]) in 0..1. Returns None for 'auto'."""
    text = str(value).strip().lstrip("#").lower()
    if text in ("auto", ""):
        return None
    if len(text) == 3:
        text = "".join(ch * 2 for ch in text)
    if len(text) != 6:
        raise ValueError(f"background_color must be a hex color like #FFFFFF or 'auto', got {value!r}")
    try:
        rgb = [int(text[i:i + 2], 16) / 255.0 for i in (0, 2, 4)]
    except ValueError:
        raise ValueError(f"background_color must be a hex color like #FFFFFF or 'auto', got {value!r}")
    return np.array(rgb, dtype=np.float32)


def estimate_background(rgb):
    """Median color of the outermost 2-pixel border."""
    h, w, _ = rgb.shape
    b = max(1, min(2, h // 2, w // 2))
    border = np.concatenate([
        rgb[:b].reshape(-1, 3),
        rgb[-b:].reshape(-1, 3),
        rgb[:, :b].reshape(-1, 3),
        rgb[:, -b:].reshape(-1, 3),
    ])
    return np.median(border, axis=0).astype(np.float32)


def subject_mask(image, mask=None, mask_convention=MASK_CONVENTIONS[0],
                 alpha_handling=ALPHA_HANDLING[0], alpha_threshold=0.5,
                 alpha_floor=0.0, tolerance=0.06, background=None):
    """
    Decide which pixels belong to the subject. Priority:
      1. an explicit mask input, if connected
      2. the alpha channel, if present and alpha handling allows it
      3. distance from the background color
    Returns a boolean (H, W) array.
    """
    h, w, c = image.shape

    if mask is not None:
        m = np.asarray(mask, dtype=np.float32)
        if m.shape != (h, w):
            raise ValueError(f"Mask size {m.shape[::-1]} does not match image size {(w, h)}.")
        if mask_convention == MASK_CONVENTIONS[0]:
            m = 1.0 - m
        return m > 0.5

    has_alpha = c == 4
    if alpha_handling == "Alpha only" and not has_alpha:
        raise ValueError("alpha_handling is 'Alpha only' but the image has no alpha channel.")

    if has_alpha and alpha_handling in ("Use alpha channel", "Alpha only"):
        alpha = image[..., 3].copy()
        alpha[alpha < alpha_floor] = 0.0
        return alpha > alpha_threshold

    rgb = image[..., :3]
    bg = background if background is not None else estimate_background(rgb)
    dist = np.abs(rgb - bg[None, None, :]).max(axis=-1)
    return dist > tolerance


def bounding_box(content, min_coverage=0.003, mode=MODES[0]):
    """
    Bounding box (x0, y0, x1, y1), exclusive end, of rows/columns whose share of
    subject pixels is above min_coverage. Returns None when nothing qualifies.
    """
    h, w = content.shape
    row_share = content.mean(axis=1)
    col_share = content.mean(axis=0)
    rows = np.where(row_share > min_coverage)[0]
    cols = np.where(col_share > min_coverage)[0]
    if rows.size == 0 or cols.size == 0:
        return None

    x0, x1 = int(cols[0]), int(cols[-1]) + 1
    y0, y1 = int(rows[0]), int(rows[-1]) + 1
    if mode == "Horizontal only":
        y0, y1 = 0, h
    elif mode == "Vertical only":
        x0, x1 = 0, w
    return x0, y0, x1, y1


def union_box(boxes):
    boxes = [b for b in boxes if b is not None]
    if not boxes:
        return None
    return (min(b[0] for b in boxes), min(b[1] for b in boxes),
            max(b[2] for b in boxes), max(b[3] for b in boxes))


def pad_box(box, padding, mode, width, height):
    """Grow the box by padding on the trimmed axes. May extend past the image edges."""
    x0, y0, x1, y1 = box
    if mode in ("Both", "Horizontal only"):
        x0, x1 = x0 - padding, x1 + padding
    if mode in ("Both", "Vertical only"):
        y0, y1 = y0 - padding, y1 + padding
    return x0, y0, x1, y1


def crop_with_fill(array, box, fill):
    """
    Crop array (H, W, C) or (H, W) to box, filling any area outside the image
    with fill (a scalar or a per-channel vector).
    """
    x0, y0, x1, y1 = box
    h, w = array.shape[:2]
    out_shape = (y1 - y0, x1 - x0) + array.shape[2:]
    out = np.empty(out_shape, dtype=array.dtype)
    out[...] = fill

    sx0, sy0 = max(x0, 0), max(y0, 0)
    sx1, sy1 = min(x1, w), min(y1, h)
    if sx1 > sx0 and sy1 > sy0:
        out[sy0 - y0:sy1 - y0, sx0 - x0:sx1 - x0] = array[sy0:sy1, sx0:sx1]
    return out


def flatten(image, background_rgb):
    """Composite RGBA over a solid color -> RGB."""
    if image.shape[-1] != 4:
        return image[..., :3]
    alpha = image[..., 3:4]
    return image[..., :3] * alpha + background_rgb[None, None, :] * (1.0 - alpha)


def center_on_canvas(array, height, width, fill):
    """Place array in the center of a (height, width) canvas filled with fill."""
    h, w = array.shape[:2]
    top = (height - h) // 2
    left = (width - w) // 2
    canvas = np.empty((height, width) + array.shape[2:], dtype=array.dtype)
    canvas[...] = fill
    canvas[top:top + h, left:left + w] = array
    return canvas


def trim_batch(images, masks=None, mode="Both", padding=32, tolerance=0.06,
               min_coverage=0.003, shared_crop_for_batch=True,
               alpha_handling="Use alpha channel", alpha_threshold=0.5,
               flatten_background=False, alpha_floor=0.0,
               background_color="#FFFFFF", mask_convention=MASK_CONVENTIONS[0]):
    """
    images: list of (H, W, C) arrays, all the same size (a ComfyUI batch).
    masks:  None or list of (H, W) arrays.
    Returns (list of trimmed images, list of subject masks), all the same size.
    """
    if mode not in MODES:
        raise ValueError(f"mode must be one of {MODES}")
    padding = max(0, int(padding))
    bg_rgb = parse_hex_color(background_color)

    contents, boxes = [], []
    for i, img in enumerate(images):
        rgb_bg = bg_rgb if bg_rgb is not None else estimate_background(img[..., :3])
        content = subject_mask(
            img,
            mask=None if masks is None else masks[i],
            mask_convention=mask_convention,
            alpha_handling=alpha_handling,
            alpha_threshold=alpha_threshold,
            alpha_floor=alpha_floor,
            tolerance=tolerance,
            background=rgb_bg,
        )
        contents.append(content)
        boxes.append(bounding_box(content, min_coverage, mode))

    h, w = images[0].shape[:2]
    full = (0, 0, w, h)
    if shared_crop_for_batch:
        shared = union_box(boxes) or full
        boxes = [shared] * len(images)
    else:
        boxes = [b or full for b in boxes]
    boxes = [pad_box(b, padding, mode, w, h) for b in boxes]

    out_images, out_masks = [], []
    for img, content, box in zip(images, contents, boxes):
        img = img.copy()
        has_alpha = img.shape[-1] == 4
        fill_rgb = bg_rgb if bg_rgb is not None else estimate_background(img[..., :3])

        if has_alpha and alpha_floor > 0:
            a = img[..., 3]
            a[a < alpha_floor] = 0.0

        if has_alpha and flatten_background:
            img = flatten(img, fill_rgb)
            has_alpha = False

        if has_alpha:
            fill = np.concatenate([fill_rgb, [0.0]]).astype(np.float32)
        else:
            img = img[..., :3]
            fill = fill_rgb

        out_images.append(crop_with_fill(img, box, fill))
        out_masks.append(crop_with_fill(content.astype(np.float32), box, 0.0))

    # Batches must share one size: center smaller crops on a canvas.
    max_h = max(o.shape[0] for o in out_images)
    max_w = max(o.shape[1] for o in out_images)
    if any(o.shape[:2] != (max_h, max_w) for o in out_images):
        canvas_images, canvas_masks = [], []
        for img, m in zip(out_images, out_masks):
            fill_rgb = bg_rgb if bg_rgb is not None else estimate_background(img[..., :3])
            if img.shape[-1] == 4:
                fill = np.concatenate([fill_rgb, [0.0]]).astype(np.float32)
            else:
                fill = fill_rgb
            canvas_images.append(center_on_canvas(img, max_h, max_w, fill))
            canvas_masks.append(center_on_canvas(m, max_h, max_w, 0.0))
        out_images, out_masks = canvas_images, canvas_masks

    return out_images, out_masks
