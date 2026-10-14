# AirbagGuy Auto Trim

A ComfyUI node that trims the empty border around a subject and adds clean padding back.
Built for character sheets: trim each view to the character before stitching.

It finds the subject in this order:

1. **Mask input**, if one is connected
2. **Alpha channel**, if the image has one (for example Qwen-Image-2.1 RGBA output)
3. **Background color**, comparing each pixel to `background_color`

## Install

**ComfyUI Manager (Install via Git URL):** open the Manager, choose *Install via Git URL*, paste `https://github.com/AirbagGuy/ComfyUI-AirbagGuyAutoTrim`, then restart ComfyUI. (If Git URL installs are blocked, set `security_level = weak` in the Manager's `config.ini`, or use the manual install below.)

**Manual:** clone this repo into `ComfyUI/custom_nodes/` and restart ComfyUI.

```
cd ComfyUI/custom_nodes
git clone https://github.com/AirbagGuy/ComfyUI-AirbagGuyAutoTrim
```

No extra Python packages are needed.

## Node: AirbagGuy Auto Trim

Category: `AirbagGuy/image`

### Inputs

| Input | Default | What it does |
|---|---|---|
| image | | Image or batch to trim. RGB or RGBA. |
| mask *(optional)* | | Explicit subject mask. Overrides alpha and color detection. |
| mode | Both | Trim all edges, left/right only, or top/bottom only. |
| padding | 32 | Pixels added around the subject on trimmed edges. Can extend past the original image; the extra area is filled with the background. |
| tolerance | 0.06 | Color mode only. How different from the background a pixel must be to count as subject. Raise it if a soft gradient background isn't trimmed. |
| min_coverage | 0.003 | Share of a row or column that must be subject before it counts. Ignores stray specks. |
| shared_crop_for_batch | true | One crop box for the whole batch, so frames stay aligned. Off: each image is cropped separately and centered on a common canvas. |
| alpha_handling | Use alpha channel | *Use alpha channel*: use alpha when present, else color. *Ignore alpha*: always use color. *Alpha only*: error if there's no alpha. |
| alpha_threshold | 0.5 | Alpha above this counts as subject. |
| flatten_background | false | Composite a transparent image over `background_color` and output RGB. |
| alpha_floor | 0.0 | Alpha below this is set to fully transparent, which cleans up faint haze. |
| background_color | #FFFFFF | Hex color, or `auto` to sample the image border. Also used for padding fill and flattening. |
| mask_convention | Mask white = background (Load Image) | How to read the mask input. Load Image masks are white where the image was transparent. |

### Outputs

| Output | What it is |
|---|---|
| image | Trimmed image (RGBA if the input had alpha and wasn't flattened, otherwise RGB). |
| mask | Subject mask for the trimmed image, white = subject. |
| width, height | Size of the trimmed image. |

## Tips

- For Qwen-Image-2.1 output with an alpha background, leave `alpha_handling` on *Use alpha channel*. Detection is pixel-exact.
- For white-background renders, the defaults work. If shadows or a faint gradient get kept, raise `tolerance`.
- If a stray dot keeps the crop too large, raise `min_coverage` a little.

## License

MIT
