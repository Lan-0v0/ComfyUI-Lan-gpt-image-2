"""
Lan-gpt-image-2: Advanced GPT Image Generation & Editing Node for ComfyUI

A comprehensive single-node plugin supporting:
  - Custom API base URL (for proxies, local endpoints, etc.)
  - Configurable model name
  - Text-to-image generation and image editing/inpainting
  - Quality, size, format, compression, background, moderation controls
  - Retry with exponential backoff
  - Custom headers for proxy authentication
  - Save-to-disk with timestamped filenames
  - Debug info output (timing, model, dimensions)
  - Batch image editing (loop through multiple reference images)
  - Environment variable fallbacks for API key and base URL
"""

import io
import os
import math
import time
import base64
import json
import re
from datetime import datetime

import numpy as np
from PIL import Image
import torch
import requests

# ---------------------------------------------------------------------------
# ComfyUI type imports (with fallbacks for standalone testing)
# ---------------------------------------------------------------------------
try:
    from comfy.utils import common_upscale
except Exception:
    common_upscale = None

try:
    from comfy.comfy_types.node_typing import IO, ComfyNodeABC, InputTypeDict
except Exception:
    class ComfyNodeABC:
        pass

    class IO:
        STRING = "STRING"
        INT = "INT"
        FLOAT = "FLOAT"
        IMAGE = "IMAGE"
        MASK = "MASK"
        COMBO = "COMBO"
        BOOLEAN = "BOOLEAN"

    InputTypeDict = dict


# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
DEFAULT_BASE_URL = "https://api.openai.com/v1"
DEFAULT_MODEL = "gpt-image-1"
MAX_INPUT_PIXELS = 1536 * 1024  # 4-megapixel cap recommended by OpenAI for gpt-image-1 edits
SUPPORTED_SIZES = [
    "auto",
    "1024x1024",
    "1024x1536",
    "1536x1024",
    "1024x1792",
    "1792x1024",
]


# ---------------------------------------------------------------------------
# Image / Tensor utility functions
# ---------------------------------------------------------------------------

def _ensure_3d_hwc(tensor: torch.Tensor) -> torch.Tensor:
    """Normalise a tensor to [H, W, C] with 3 or 4 channels."""
    t = tensor.cpu().detach().float()
    if t.ndim == 4:
        # [B, H, W, C] -> take first batch
        t = t[0]
    if t.ndim == 3:
        # [C, H, W] -> [H, W, C]
        if t.shape[0] in (3, 4) and t.shape[-1] not in (3, 4):
            t = t.permute(1, 2, 0)
    if t.ndim == 2:
        t = t.unsqueeze(-1).repeat(1, 1, 3)
    if t.ndim != 3 or t.shape[-1] not in (3, 4):
        raise ValueError(
            f"Cannot normalise tensor with shape {tensor.shape} -> {t.shape}. "
            f"Expected [H, W, 3] or [H, W, 4]."
        )
    return t


def downscale_image(tensor_hwc: torch.Tensor, max_pixels: int = MAX_INPUT_PIXELS) -> torch.Tensor:
    """Downscale an [H, W, C] tensor if it exceeds *max_pixels*, preserving aspect ratio."""
    h, w = tensor_hwc.shape[0], tensor_hwc.shape[1]
    if h * w <= max_pixels:
        return tensor_hwc
    if common_upscale is None:
        # Fallback: simple PIL resize
        ratio = math.sqrt(max_pixels / (h * w))
        nw, nh = round(w * ratio), round(h * ratio)
        arr = (tensor_hwc.numpy() * 255).clip(0, 255).astype(np.uint8)
        pil = Image.fromarray(arr if arr.shape[-1] == 3 else arr)
        pil = pil.resize((nw, nh), Image.LANCZOS)
        arr = np.array(pil).astype(np.float32) / 255.0
        return torch.from_numpy(arr)
    ratio = math.sqrt(max_pixels / (h * w))
    nw, nh = round(w * ratio), round(h * ratio)
    # common_upscale expects [B, C, H, W]
    batched = tensor_hwc.unsqueeze(0).movedim(-1, 1)  # [1, C, H, W]
    upscaled = common_upscale(batched, nw, nh, "lanczos", "disabled")
    return upscaled.squeeze(0).movedim(0, -1)  # back to [H, W, C]


def tensor_to_png_bytes(image_tensor: torch.Tensor) -> io.BytesIO:
    """Convert a ComfyUI image tensor (any common shape) to PNG bytes."""
    t = _ensure_3d_hwc(image_tensor)
    t = downscale_image(t)
    arr = (t.numpy() * 255).clip(0, 255).astype(np.uint8)
    if arr.shape[-1] == 4:
        pil = Image.fromarray(arr, "RGBA")
    else:
        pil = Image.fromarray(arr, "RGB")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf


def mask_to_png_bytes(mask_tensor: torch.Tensor, target_hw) -> io.BytesIO:
    """Convert a mask tensor to a PNG where transparent = edit area, opaque = keep."""
    h, w = target_hw
    mask = mask_tensor.cpu().detach().float()
    # Squeeze to 2D
    if mask.ndim == 3:
        if mask.shape[-1] == 1:
            mask = mask.squeeze(-1)
        elif mask.shape[0] == 1:
            mask = mask.squeeze(0)
        else:
            mask = mask.mean(dim=0) if mask.ndim == 3 else mask[0]
    if mask.ndim > 2:
        mask = mask.view(-1, h, w)[0] if mask.numel() >= h * w else mask.reshape(h, w)

    # Resize to match image if needed
    if mask.shape != (h, w):
        if common_upscale is not None:
            m = mask.unsqueeze(0).unsqueeze(0)
            m = common_upscale(m, w, h, "nearest", "disabled")
            mask = m.squeeze(0).squeeze(0)
        else:
            arr = (mask.numpy() * 255).clip(0, 255).astype(np.uint8)
            pil = Image.fromarray(arr, "L").resize((w, h), Image.NEAREST)
            mask = torch.from_numpy(np.array(pil).astype(np.float32) / 255.0)

    # OpenAI mask convention: transparent (alpha=0) = area to edit
    rgba = np.zeros((h, w, 4), dtype=np.uint8)
    # Where mask > 0.5 -> keep (opaque, alpha=255); else -> edit (transparent, alpha=0)
    rgba[mask.numpy() > 0.5, 3] = 255
    pil = Image.fromarray(rgba, "RGBA")
    buf = io.BytesIO()
    pil.save(buf, format="PNG")
    buf.seek(0)
    return buf


def process_api_response(response_json: dict) -> torch.Tensor:
    """Parse the JSON response from the image API and return a [B, H, W, C] tensor."""
    if "data" not in response_json or not response_json["data"]:
        err = response_json.get("error", {})
        if isinstance(err, dict):
            msg = err.get("message", str(err))
        else:
            msg = str(err)
        raise RuntimeError(f"API returned no image data. Error: {msg}")

    tensors = []
    for item in response_json["data"]:
        b64 = item.get("b64_json")
        url = item.get("url")
        pil_img = None

        if b64:
            raw = base64.b64decode(b64)
            pil_img = Image.open(io.BytesIO(raw))
        elif url:
            resp = requests.get(url, timeout=60)
            resp.raise_for_status()
            pil_img = Image.open(io.BytesIO(resp.content))

        if pil_img is not None:
            pil_img = pil_img.convert("RGBA")
            arr = np.array(pil_img).astype(np.float32) / 255.0
            tensors.append(torch.from_numpy(arr))

    if not tensors:
        raise RuntimeError("Failed to extract any images from the API response.")

    return torch.stack(tensors, dim=0)


def save_images_to_disk(image_batch: torch.Tensor, output_dir: str, prefix: str = "lan_gpt_image") -> list:
    """Save each image in a [B, H, W, C] batch to disk. Returns list of file paths."""
    os.makedirs(output_dir, exist_ok=True)
    ts = datetime.now().strftime("%Y%m%d_%H%M%S")
    paths = []
    for i in range(image_batch.shape[0]):
        t = image_batch[i].cpu().detach()
        arr = (t.numpy() * 255).clip(0, 255).astype(np.uint8)
        mode = "RGBA" if arr.shape[-1] == 4 else "RGB"
        pil = Image.fromarray(arr, mode)
        fname = f"{prefix}_{ts}_{i:03d}.png"
        fpath = os.path.join(output_dir, fname)
        pil.save(fpath, format="PNG")
        paths.append(fpath)
    return paths


def normalise_base_url(url: str) -> str:
    """Strip trailing slashes and ensure no double slashes in the path."""
    url = url.strip()
    if not url:
        return DEFAULT_BASE_URL
    # Remove trailing slashes but keep the scheme
    while url.endswith("/"):
        url = url[:-1]
    return url


def parse_extra_headers(raw: str) -> dict:
    """Parse a JSON string of extra headers. Returns empty dict on failure."""
    if not raw or not raw.strip():
        return {}
    try:
        headers = json.loads(raw)
        if not isinstance(headers, dict):
            raise ValueError("Extra headers must be a JSON object.")
        return headers
    except (json.JSONDecodeError, ValueError) as e:
        raise ValueError(f"Invalid extra_headers JSON: {e}") from e


# ---------------------------------------------------------------------------
# Node
# ---------------------------------------------------------------------------

class LanGPTImage2(ComfyNodeABC):
    """
    Lan-gpt-image-2 — Advanced GPT Image Generation & Editing

    Generates or edits images through any OpenAI-compatible image API endpoint.
    Supports custom base URLs (proxies, local servers, alternative providers),
    full parameter control, retry logic, custom headers, and save-to-disk.
    """

    def __init__(self):
        pass

    @classmethod
    def INPUT_TYPES(cls) -> InputTypeDict:
        return {
            "required": {
                "prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Text prompt describing the image to generate or the edit to apply.",
                    },
                ),
                "api_key": (
                    IO.STRING,
                    {
                        "multiline": False,
                        "default": "",
                        "password": True,
                        "tooltip": "API key. Leave blank to use the OPENAI_API_KEY environment variable.",
                    },
                ),
                "base_url": (
                    IO.STRING,
                    {
                        "multiline": False,
                        "default": DEFAULT_BASE_URL,
                        "tooltip": (
                            "API base URL. Use https://api.openai.com/v1 for official OpenAI, "
                            "or a custom URL like http://localhost:8317/v1 for proxies."
                        ),
                    },
                ),
            },
            "optional": {
                "model": (
                    IO.STRING,
                    {
                        "default": DEFAULT_MODEL,
                        "tooltip": "Model name. Default: gpt-image-1. Change if your proxy/provider uses a different name.",
                    },
                ),
                "quality": (
                    IO.COMBO,
                    {
                        "options": ["auto", "low", "medium", "high"],
                        "default": "auto",
                        "tooltip": "Image quality. Affects cost and generation time.",
                    },
                ),
                "size": (
                    IO.COMBO,
                    {
                        "options": SUPPORTED_SIZES,
                        "default": "auto",
                        "tooltip": "Output image dimensions.",
                    },
                ),
                "n": (
                    IO.INT,
                    {
                        "default": 1,
                        "min": 1,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Number of images to generate per request.",
                    },
                ),
                "background": (
                    IO.COMBO,
                    {
                        "options": ["opaque", "transparent"],
                        "default": "opaque",
                        "tooltip": "Whether the generated image has a transparent background.",
                    },
                ),
                "moderation": (
                    IO.COMBO,
                    {
                        "options": ["auto", "low", "none"],
                        "default": "auto",
                        "tooltip": (
                            "Content moderation strictness. "
                            "'auto' = standard filtering, 'low' = less restrictive, "
                            "'none' = disable moderation (only works if the API/proxy supports it; "
                            "enable auto_fallback_moderation to retry with 'low' if rejected)."
                        ),
                    },
                ),
                "auto_fallback_moderation": (
                    IO.BOOLEAN,
                    {
                        "default": True,
                        "tooltip": (
                            "If the API rejects the selected moderation level (e.g. 'none'), "
                            "automatically retry with 'low' mode instead of failing."
                        ),
                    },
                ),
                "output_format": (
                    IO.COMBO,
                    {
                        "options": ["png", "jpeg", "webp"],
                        "default": "png",
                        "tooltip": "Format of the returned image data from the API.",
                    },
                ),
                "output_compression": (
                    IO.INT,
                    {
                        "default": 85,
                        "min": 0,
                        "max": 100,
                        "step": 1,
                        "tooltip": "Compression level (0-100) for jpeg/webp output. Ignored for png.",
                    },
                ),
                "seed": (
                    IO.INT,
                    {
                        "default": 0,
                        "min": 0,
                        "max": 0xFFFFFFFFFFFFFFFF,
                        "step": 1,
                        "display": "number",
                        "tooltip": "Random seed. May or may not be honoured by the API.",
                    },
                ),
                "negative_prompt": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": "Negative prompt — describes what to avoid. (Sent only if the API supports it.)",
                    },
                ),
                "image": (
                    IO.IMAGE,
                    {
                        "tooltip": "Reference image for editing/inpainting. Connect with 'mask' to edit a specific area.",
                    },
                ),
                "mask": (
                    IO.MASK,
                    {
                        "tooltip": "Inpainting mask. White (value > 0.5) = keep, black = edit area. Must be used with 'image'.",
                    },
                ),
                "timeout": (
                    IO.INT,
                    {
                        "default": 120,
                        "min": 10,
                        "max": 600,
                        "step": 1,
                        "tooltip": "HTTP request timeout in seconds.",
                    },
                ),
                "max_retries": (
                    IO.INT,
                    {
                        "default": 3,
                        "min": 0,
                        "max": 10,
                        "step": 1,
                        "tooltip": "Maximum number of retry attempts on transient failures.",
                    },
                ),
                "retry_delay": (
                    IO.FLOAT,
                    {
                        "default": 2.0,
                        "min": 0.0,
                        "max": 60.0,
                        "step": 0.5,
                        "tooltip": "Base delay in seconds between retries (exponential backoff applied).",
                    },
                ),
                "extra_headers": (
                    IO.STRING,
                    {
                        "multiline": True,
                        "default": "",
                        "tooltip": 'Additional HTTP headers as JSON, e.g. {"X-Custom-Header": "value"}. Useful for proxy auth.',
                    },
                ),
                "save_to_disk": (
                    IO.BOOLEAN,
                    {
                        "default": False,
                        "tooltip": "If True, save generated images to output_dir.",
                    },
                ),
                "output_dir": (
                    IO.STRING,
                    {
                        "multiline": False,
                        "default": "lan_gpt_image_output",
                        "tooltip": "Directory name for saved images. Created under ComfyUI output/ if relative.",
                    },
                ),
            },
        }

    RETURN_TYPES = (IO.IMAGE, IO.STRING)
    RETURN_NAMES = ("images", "info")
    FUNCTION = "generate"
    CATEGORY = "Lan/gpt-image"
    OUTPUT_IS_LIST = (False, False)
    DESCRIPTION = (
        "Advanced GPT image generation & editing node with custom API endpoint support. "
        "Use any OpenAI-compatible image API by setting base_url (e.g. http://localhost:8317/v1 for a local proxy). "
        "Supports text-to-image, image editing, inpainting, quality/size/format controls, retry logic, and more."
    )
    API_NODE = False

    # -----------------------------------------------------------------------
    # Main execution
    # -----------------------------------------------------------------------

    def generate(
        self,
        prompt: str,
        api_key: str,
        base_url: str,
        model: str = DEFAULT_MODEL,
        quality: str = "auto",
        size: str = "auto",
        n: int = 1,
        background: str = "opaque",
        moderation: str = "auto",
        auto_fallback_moderation: bool = True,
        output_format: str = "png",
        output_compression: int = 85,
        seed: int = 0,
        negative_prompt: str = "",
        image=None,
        mask=None,
        timeout: int = 120,
        max_retries: int = 3,
        retry_delay: float = 2.0,
        extra_headers: str = "",
        save_to_disk: bool = False,
        output_dir: str = "lan_gpt_image_output",
    ):
        # --- Resolve API key ---
        resolved_key = (api_key or "").strip()
        if not resolved_key:
            resolved_key = os.environ.get("OPENAI_API_KEY", "").strip()
        if not resolved_key:
            raise ValueError(
                "An API key is required. Provide it via the 'api_key' input "
                "or set the OPENAI_API_KEY environment variable."
            )

        # --- Resolve base URL ---
        resolved_base = normalise_base_url(base_url)
        # Allow env override when the input is left at default and env is set
        env_base = os.environ.get("OPENAI_BASE_URL", "").strip()
        if not env_base and os.environ.get("OPENAI_API_BASE", "").strip():
            env_base = os.environ.get("OPENAI_API_BASE", "").strip()
        if env_base and resolved_base == DEFAULT_BASE_URL:
            resolved_base = normalise_base_url(env_base)

        # --- Parse extra headers ---
        extra_hdrs = parse_extra_headers(extra_headers)

        # --- Validate prompt ---
        if not prompt or not prompt.strip():
            raise ValueError("Prompt cannot be empty.")

        # --- Determine mode (generation vs. edit) ---
        has_image = image is not None
        has_mask = mask is not None

        if has_mask and not has_image:
            raise ValueError("'mask' input requires 'image' to be connected as well.")

        is_edit = has_image  # image present -> edit endpoint (mask optional for whole-image edit)

        # --- Build common data payload ---
        data = {
            "model": model,
            "prompt": prompt,
            "n": n,
        }
        if size and size != "auto":
            data["size"] = size
        elif size == "auto":
            data["size"] = "auto"
        if quality:
            data["quality"] = quality
        if background:
            data["background"] = background
        if moderation:
            data["moderation"] = moderation
        if output_format and output_format != "png":
            data["output_format"] = output_format
            data["output_compression"] = output_compression
        if negative_prompt and negative_prompt.strip():
            data["negative_prompt"] = negative_prompt.strip()
        if seed:
            data["seed"] = seed

        # --- Build headers ---
        headers = {
            "Authorization": f"Bearer {resolved_key}",
        }
        headers.update(extra_hdrs)

        # --- Determine endpoint ---
        endpoint = f"{resolved_base}/images/edits" if is_edit else f"{resolved_base}/images/generations"

        # --- Execute request(s) with retry ---
        info_lines = []
        info_lines.append(f"Endpoint: {endpoint}")
        info_lines.append(f"Model: {model}")
        info_lines.append(f"Mode: {'edit' if is_edit else 'generation'}")
        info_lines.append(f"Quality: {quality} | Size: {size} | N: {n}")
        info_lines.append(f"Background: {background} | Moderation: {moderation} (auto-fallback: {auto_fallback_moderation})")
        info_lines.append(f"Output format: {output_format} (compression: {output_compression})")

        t0 = time.time()
        all_tensors = []

        if is_edit:
            # --- Image editing ---
            batch_size = image.shape[0] if hasattr(image, "shape") else 1
            info_lines.append(f"Input batch size: {batch_size}")

            for batch_idx in range(batch_size):
                single_img = image[batch_idx] if batch_size > 1 else image
                # If image is [B, H, W, C], extract single
                if single_img.ndim == 4:
                    single_img = single_img[0]

                files = {}
                try:
                    img_buf = tensor_to_png_bytes(single_img)
                    files["image"] = ("image.png", img_buf, "image/png")

                    if has_mask:
                        # Extract matching mask
                        single_mask = mask
                        if hasattr(mask, "shape") and mask.ndim >= 2:
                            if mask.ndim == 3 and mask.shape[0] == batch_size:
                                single_mask = mask[batch_idx]
                            elif mask.ndim == 3 and batch_size == 1:
                                single_mask = mask[0] if mask.shape[0] == 1 else mask

                        img_hw = (single_img.shape[0], single_img.shape[1]) if single_img.ndim == 3 else (1, 1)
                        mask_buf = mask_to_png_bytes(single_mask, img_hw)
                        files["mask"] = ("mask.png", mask_buf, "image/png")

                    resp_json = self._do_request_with_moderation_fallback(
                        endpoint, headers, data, files, timeout, max_retries, retry_delay,
                        info_lines, batch_idx, auto_fallback_moderation, moderation,
                    )
                    batch_tensors = process_api_response(resp_json)
                    all_tensors.append(batch_tensors)
                except Exception as e:
                    info_lines.append(f"ERROR on batch {batch_idx}: {e}")
                    raise

        else:
            # --- Text-to-image generation ---
            resp_json = self._do_request_with_moderation_fallback(
                endpoint, headers, data, None, timeout, max_retries, retry_delay,
                info_lines, 0, auto_fallback_moderation, moderation,
            )
            batch_tensors = process_api_response(resp_json)
            all_tensors.append(batch_tensors)

        elapsed = time.time() - t0

        # --- Combine all result tensors ---
        if len(all_tensors) == 1:
            result = all_tensors[0]
        else:
            # Concatenate along batch dimension
            result = torch.cat(all_tensors, dim=0)

        # --- Save to disk ---
        if save_to_disk:
            # Resolve output directory
            out_path = output_dir
            if not os.path.isabs(out_path):
                # Try to use ComfyUI output directory
                try:
                    import folder_paths
                    comfy_out = folder_paths.get_output_directory()
                    out_path = os.path.join(comfy_out, out_path)
                except Exception:
                    out_path = os.path.abspath(out_path)
            saved = save_images_to_disk(result, out_path)
            info_lines.append(f"Saved {len(saved)} image(s) to: {out_path}")
            for p in saved:
                info_lines.append(f"  - {p}")

        # --- Build info string ---
        info_lines.append(f"Total elapsed: {elapsed:.2f}s")
        info_lines.append(f"Output tensor shape: {list(result.shape)}")
        info_str = "\n".join(info_lines)

        print(f"[Lan-gpt-image-2] {info_str}")
        return (result, info_str)

    # -----------------------------------------------------------------------
    # Retry-aware request helper
    # -----------------------------------------------------------------------

    def _do_request_with_retry(
        self,
        endpoint: str,
        headers: dict,
        data: dict,
        files: dict | None,
        timeout: int,
        max_retries: int,
        retry_delay: float,
        info_lines: list,
        batch_idx: int,
    ) -> dict:
        """Send the HTTP request, retrying on transient failures with exponential backoff."""
        attempt = 0
        last_error = None

        while attempt <= max_retries:
            attempt += 1
            try:
                if files:
                    # Multipart form (image editing)
                    # Don't set Content-Type for multipart; requests handles it
                    hdrs = {k: v for k, v in headers.items()}
                    resp = requests.post(endpoint, headers=hdrs, data=data, files=files, timeout=timeout)
                else:
                    hdrs = dict(headers)
                    hdrs["Content-Type"] = "application/json"
                    resp = requests.post(endpoint, headers=hdrs, json=data, timeout=timeout)

                if resp.status_code == 200:
                    return resp.json()

                # Non-200: check if retryable
                error_body = ""
                try:
                    error_body = resp.text
                except Exception:
                    pass

                # 429 (rate limit) and 5xx are retryable
                retryable = resp.status_code == 429 or resp.status_code >= 500
                if not retryable or attempt > max_retries:
                    # Parse error message
                    err_msg = error_body
                    try:
                        err_json = resp.json()
                        if "error" in err_json:
                            err_msg = err_json["error"].get("message", error_body)
                    except Exception:
                        pass
                    raise RuntimeError(
                        f"API request failed (HTTP {resp.status_code}){' on batch ' + str(batch_idx) if batch_idx else ''}: {err_msg}"
                    )

                last_error = f"HTTP {resp.status_code}: {error_body[:200]}"
                info_lines.append(
                    f"Attempt {attempt}/{max_retries + 1} failed (HTTP {resp.status_code}). Retrying in {retry_delay * attempt:.1f}s..."
                )
                time.sleep(retry_delay * attempt)  # exponential backoff

            except requests.exceptions.ConnectionError as e:
                last_error = str(e)
                if attempt > max_retries:
                    raise RuntimeError(
                        f"Connection error after {attempt} attempts: {e}"
                    ) from e
                info_lines.append(
                    f"Connection error on attempt {attempt}/{max_retries + 1}. Retrying in {retry_delay * attempt:.1f}s..."
                )
                time.sleep(retry_delay * attempt)

            except requests.exceptions.Timeout as e:
                last_error = str(e)
                if attempt > max_retries:
                    raise RuntimeError(
                        f"Request timed out after {timeout}s after {attempt} attempts."
                    ) from e
                info_lines.append(
                    f"Timeout on attempt {attempt}/{max_retries + 1}. Retrying in {retry_delay * attempt:.1f}s..."
                )
                time.sleep(retry_delay * attempt)

            except RuntimeError:
                raise  # Non-retryable API error

            except Exception as e:
                last_error = str(e)
                if attempt > max_retries:
                    raise RuntimeError(
                        f"Unexpected error after {attempt} attempts: {e}"
                    ) from e
                info_lines.append(
                    f"Error on attempt {attempt}/{max_retries + 1}: {e}. Retrying..."
                )
                time.sleep(retry_delay * attempt)

        raise RuntimeError(f"All {max_retries + 1} attempts failed. Last error: {last_error}")

    # -----------------------------------------------------------------------
    # Moderation fallback wrapper
    # -----------------------------------------------------------------------

    @staticmethod
    def _is_moderation_error(error: RuntimeError) -> bool:
        """Check if an API error is related to an unsupported moderation value."""
        err_str = str(error).lower()
        # Match patterns like "invalid value for 'moderation'" or "moderation" mentions
        if "moderation" in err_str:
            return True
        # Also match generic "invalid value" 400 errors that might be about moderation
        if "400" in err_str and "invalid" in err_str and "value" in err_str:
            return True
        return False

    def _do_request_with_moderation_fallback(
        self,
        endpoint: str,
        headers: dict,
        data: dict,
        files: dict | None,
        timeout: int,
        max_retries: int,
        retry_delay: float,
        info_lines: list,
        batch_idx: int,
        auto_fallback: bool,
        original_moderation: str,
    ) -> dict:
        """
        Wrap _do_request_with_retry with automatic moderation fallback.

        If the API rejects the selected moderation level (e.g. 'none') and
        auto_fallback is True, retry once with moderation='low'.
        """
        try:
            return self._do_request_with_retry(
                endpoint, headers, data, files, timeout, max_retries, retry_delay, info_lines, batch_idx
            )
        except RuntimeError as e:
            if auto_fallback and original_moderation != "low" and self._is_moderation_error(e):
                info_lines.append(
                    f"Moderation '{original_moderation}' was rejected by the API. "
                    f"Auto-falling back to 'low' mode and retrying..."
                )
                print(f"[Lan-gpt-image-2] Moderation fallback: '{original_moderation}' -> 'low'")
                data["moderation"] = "low"
                # Seek file objects back to start for retry (edit mode)
                if files:
                    for val in files.values():
                        if isinstance(val, tuple) and len(val) >= 2:
                            file_obj = val[1]
                            if hasattr(file_obj, "seek"):
                                file_obj.seek(0)
                return self._do_request_with_retry(
                    endpoint, headers, data, files, timeout, max_retries, retry_delay, info_lines, batch_idx
                )
            raise


# ---------------------------------------------------------------------------
# ComfyUI registration
# ---------------------------------------------------------------------------

NODE_CLASS_MAPPINGS = {
    "Lan-gpt-image-2": LanGPTImage2,
}

NODE_DISPLAY_NAME_MAPPINGS = {
    "Lan-gpt-image-2": "Lan-gpt-image-2",
}

__all__ = ["NODE_CLASS_MAPPINGS", "NODE_DISPLAY_NAME_MAPPINGS"]
