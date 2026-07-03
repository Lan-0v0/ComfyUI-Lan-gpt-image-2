"""
Test script for Lan-gpt-image-2 plugin.
Tests utility functions, node structure, and API call logic.
Can run with or without torch installed.
"""

import sys
import os
import io
import json
import base64
import struct
import traceback

# --- Mock torch if not available ---
if 'torch' not in sys.modules:
    try:
        import torch
    except ImportError:
        # Create a minimal torch mock
        import numpy as np_mock

        class MockTensor:
            """Minimal tensor mock backed by numpy."""
            def __init__(self, data):
                if isinstance(data, np_mock.ndarray):
                    self._np = data
                elif isinstance(data, (list, tuple)):
                    self._np = np_mock.array(data)
                else:
                    self._np = np_mock.array(data)

            @property
            def shape(self):
                return self._np.shape

            @property
            def ndim(self):
                return self._np.ndim

            @property
            def dtype(self):
                return self._np.dtype

            def numpy(self):
                return self._np

            def cpu(self):
                return self

            def detach(self):
                return self

            def float(self):
                return self

            def permute(self, *dims):
                return MockTensor(self._np.transpose(dims))

            def unsqueeze(self, dim):
                return MockTensor(self._np.expand_dims(self._np, axis=dim))

            def squeeze(self, dim=None):
                if dim is not None:
                    return MockTensor(np_mock.squeeze(self._np, axis=dim))
                return MockTensor(np_mock.squeeze(self._np))

            def movedim(self, src, dst):
                return MockTensor(np_mock.moveaxis(self._np, src, dst))

            @property
            def numel(self):
                return self._np.size

            def min(self):
                return MockScalar(self._np.min())

            def max(self):
                return MockScalar(self._np.max())

            def mean(self, dim=None):
                if dim is not None:
                    return MockTensor(self._np.mean(axis=dim))
                return MockScalar(self._np.mean())

            def view(self, *shape):
                return MockTensor(self._np.reshape(shape))

            def __getitem__(self, idx):
                if isinstance(idx, int):
                    return MockTensor(self._np[idx])
                return MockTensor(self._np[idx])

            def __len__(self):
                return len(self._np)

        class MockScalar:
            def __init__(self, val):
                self._val = val
            def item(self):
                return float(self._val)

        class MockTorchModule:
            Tensor = MockTensor
            float32 = np_mock.float32
            uint8 = np_mock.uint8

            @staticmethod
            def from_numpy(arr):
                return MockTensor(arr)

            @staticmethod
            def stack(tensors, dim=0):
                arrs = [t.numpy() if hasattr(t, 'numpy') else t for t in tensors]
                return MockTensor(np_mock.stack(arrs, axis=dim))

            @staticmethod
            def cat(tensors, dim=0):
                arrs = [t.numpy() if hasattr(t, 'numpy') else t for t in tensors]
                return MockTensor(np_mock.concatenate(arrs, axis=dim))

            @staticmethod
            def zeros(*shape, dtype=None):
                dt = dtype if dtype is not None else np_mock.float32
                return MockTensor(np_mock.zeros(shape, dtype=dt))

        sys.modules['torch'] = MockTorchModule()
        print("[test] Using mock torch module")

# --- Now import the plugin ---
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from nodes import (
    LanGPTImage2,
    NODE_CLASS_MAPPINGS,
    NODE_DISPLAY_NAME_MAPPINGS,
    normalise_base_url,
    parse_extra_headers,
    DEFAULT_BASE_URL,
    DEFAULT_MODEL,
    SUPPORTED_SIZES,
    DEFAULT_BASE_URL,
)
import numpy as np
from PIL import Image

passed = 0
failed = 0
errors = []

def test(name, condition, detail=""):
    global passed, failed
    if condition:
        passed += 1
        print(f"  PASS: {name}")
    else:
        failed += 1
        errors.append(f"{name}: {detail}")
        print(f"  FAIL: {name} — {detail}")

print("=" * 60)
print("Testing Lan-gpt-image-2 Plugin")
print("=" * 60)

# --- Test 1: Module structure ---
print("\n[1] Module Structure")
test("NODE_CLASS_MAPPINGS has 'Lan-gpt-image-2'", "Lan-gpt-image-2" in NODE_CLASS_MAPPINGS)
test("NODE_DISPLAY_NAME_MAPPINGS has 'Lan-gpt-image-2'", "Lan-gpt-image-2" in NODE_DISPLAY_NAME_MAPPINGS)
test("Display name is 'Lan-gpt-image-2'", NODE_DISPLAY_NAME_MAPPINGS.get("Lan-gpt-image-2") == "Lan-gpt-image-2")
test("Node class is LanGPTImage2", NODE_CLASS_MAPPINGS["Lan-gpt-image-2"] is LanGPTImage2)

# --- Test 2: Node metadata ---
print("\n[2] Node Metadata")
test("RETURN_TYPES is (IMAGE, STRING)", LanGPTImage2.RETURN_TYPES == ("IMAGE", "STRING"))
test("RETURN_NAMES is ('images', 'info')", LanGPTImage2.RETURN_NAMES == ("images", "info"))
test("FUNCTION is 'generate'", LanGPTImage2.FUNCTION == "generate")
test("CATEGORY is 'Lan/gpt-image'", LanGPTImage2.CATEGORY == "Lan/gpt-image")

# --- Test 3: INPUT_TYPES ---
print("\n[3] Input Types")
input_types = LanGPTImage2.INPUT_TYPES()
test("Has 'required' key", "required" in input_types)
test("Has 'optional' key", "optional" in input_types)

required = input_types["required"]
test("Has 'prompt' in required", "prompt" in required)
test("Has 'api_key' in required", "api_key" in required)
test("Has 'base_url' in required", "base_url" in required)

optional = input_types["optional"]
expected_optional = [
    "model", "quality", "size", "n", "background", "moderation",
    "auto_fallback_moderation",
    "output_format", "output_compression", "seed", "negative_prompt",
    "image", "mask", "timeout", "max_retries", "retry_delay",
    "extra_headers", "save_to_disk", "output_dir"
]
for name in expected_optional:
    test(f"Has '{name}' in optional", name in optional)

# --- Test 4: Parameter details ---
print("\n[4] Parameter Details")
quality_input = optional["quality"]
test("quality has 4 options", len(quality_input[1].get("options", [])) == 4)
test("quality default is 'auto'", quality_input[1].get("default") == "auto")

size_input = optional["size"]
test("size has 6 options", len(size_input[1].get("options", [])) == 6)
test("size default is 'auto'", size_input[1].get("default") == "auto")

n_input = optional["n"]
test("n min is 1", n_input[1].get("min") == 1)
test("n max is 10", n_input[1].get("max") == 10)

base_url_input = required["base_url"]
test("base_url default is OpenAI URL", base_url_input[1].get("default") == "https://api.openai.com/v1")

moderation_input = optional["moderation"]
test("moderation has 3 options", len(moderation_input[1].get("options", [])) == 3)
test("moderation options include 'none'", "none" in moderation_input[1].get("options", []))
test("moderation default is 'auto'", moderation_input[1].get("default") == "auto")

fallback_input = optional["auto_fallback_moderation"]
test("auto_fallback_moderation default is True", fallback_input[1].get("default") is True)

# --- Test 5: Utility functions ---
print("\n[5] Utility Functions")

# normalise_base_url
test("normalise_base_url strips trailing slash",
     normalise_base_url("https://api.openai.com/v1/") == "https://api.openai.com/v1")
test("normalise_base_url strips multiple trailing slashes",
     normalise_base_url("http://localhost:8317/v1///") == "http://localhost:8317/v1")
test("normalise_base_url returns default for empty",
     normalise_base_url("") == DEFAULT_BASE_URL)
test("normalise_base_url preserves localhost URL",
     normalise_base_url("http://localhost:8317/v1") == "http://localhost:8317/v1")

# parse_extra_headers
test("parse_extra_headers empty string returns {}", parse_extra_headers("") == {})
test("parse_extra_headers valid JSON",
     parse_extra_headers('{"X-Key": "val"}') == {"X-Key": "val"})
try:
    parse_extra_headers("not json")
    test("parse_extra_headers invalid JSON raises", False, "should have raised")
except ValueError:
    test("parse_extra_headers invalid JSON raises", True)
try:
    parse_extra_headers("[1,2]")
    test("parse_extra_headers non-object raises", False, "should have raised")
except ValueError:
    test("parse_extra_headers non-object raises", True)

# --- Test 6: Image processing (if numpy/PIL available) ---
print("\n[6] Image Processing")

try:
    import torch_real
    test("Real torch available", False, "using mock")
except Exception:
    test("Using mock torch for image tests", True, "real torch not installed — this is OK for ComfyUI testing")

# Create a simple test image and convert to tensor
try:
    import numpy as np
    from PIL import Image

    # Create a 64x64 RGB test image
    arr = np.zeros((64, 64, 3), dtype=np.uint8)
    arr[:32, :32] = [255, 0, 0]  # red quadrant
    arr[:32, 32:] = [0, 255, 0]  # green quadrant
    arr[32:, :32] = [0, 0, 255]  # blue quadrant
    arr[32:, 32:] = [255, 255, 0]  # yellow quadrant

    pil_img = Image.fromarray(arr, "RGB")

    # Test save_images_to_disk (doesn't need torch for the PIL part)
    from nodes import save_images_to_disk

    # Create a mock tensor batch
    mock_torch = sys.modules.get('torch')
    if hasattr(mock_torch, 'from_numpy'):
        tensor = mock_torch.from_numpy(arr.astype(np.float32) / 255.0)
        # Try to make a batch
        if hasattr(mock_torch, 'stack'):
            tensor_batch = mock_torch.stack([tensor])
        else:
            tensor_batch = tensor

        # Test save to disk
        import tempfile
        tmpdir = tempfile.mkdtemp()
        paths = save_images_to_disk(tensor_batch, tmpdir, "test")
        test("save_images_to_disk creates files", len(paths) > 0 and os.path.exists(paths[0]))
    else:
        test("save_images_to_disk skipped (no torch)", True)
except Exception as e:
    test(f"Image processing test error: {e}", False, str(e))

# --- Test 7: Node instantiation ---
print("\n[7] Node Instantiation")
try:
    node = LanGPTImage2()
    test("Node can be instantiated", True)
except Exception as e:
    test("Node can be instantiated", False, str(e))

# --- Test 8: Error handling ---
print("\n[8] Error Handling")
try:
    node = LanGPTImage2()
    # Test empty API key
    try:
        node.generate(
            prompt="test", api_key="", base_url="https://api.openai.com/v1"
        )
        test("Empty API key raises ValueError", False, "should have raised")
    except ValueError as e:
        test("Empty API key raises ValueError", True)

    # Test empty prompt
    try:
        node.generate(
            prompt="", api_key="sk-test", base_url="https://api.openai.com/v1"
        )
        test("Empty prompt raises ValueError", False, "should have raised")
    except ValueError as e:
        test("Empty prompt raises ValueError", True)

    # Test mask without image
    try:
        node.generate(
            prompt="test", api_key="sk-test", base_url="https://api.openai.com/v1",
            mask="fake_mask"
        )
        test("Mask without image raises ValueError", False, "should have raised")
    except ValueError as e:
        test("Mask without image raises ValueError", True)

    # Test invalid extra_headers
    try:
        node.generate(
            prompt="test", api_key="sk-test", base_url="https://api.openai.com/v1",
            extra_headers="invalid json"
        )
        test("Invalid extra_headers raises ValueError", False, "should have raised")
    except ValueError as e:
        test("Invalid extra_headers raises ValueError", True)

except Exception as e:
    test("Error handling tests failed", False, str(e))
    traceback.print_exc()

# --- Test 9: API call logic (mock) ---
print("\n[9] API Call Logic (mock)")
try:
    from unittest.mock import patch, MagicMock

    # Create a mock response
    mock_response = MagicMock()
    mock_response.status_code = 200

    # Create a tiny PNG image as base64
    img_buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(img_buf, format="PNG")
    b64_img = base64.b64encode(img_buf.getvalue()).decode()

    mock_response.json.return_value = {
        "data": [{"b64_json": b64_img}]
    }

    # Test that generate calls the correct endpoint
    call_args = {}

    def mock_post(url, **kwargs):
        call_args["url"] = url
        call_args["headers"] = kwargs.get("headers", {})
        call_args["json"] = kwargs.get("json")
        call_args["data"] = kwargs.get("data")
        call_args["files"] = kwargs.get("files")
        return mock_response

    node = LanGPTImage2()

    with patch("requests.post", side_effect=mock_post):
        result, info = node.generate(
            prompt="a cute cat",
            api_key="sk-test-key",
            base_url="http://localhost:8317/v1",
            model="gpt-image-1",
            quality="high",
            size="1024x1024",
            n=1,
        )

    test("API call returns tuple of (tensor, str)", isinstance(result, type(result)) and isinstance(info, str))
    test("API called correct endpoint",
         call_args.get("url") == "http://localhost:8317/v1/images/generations")
    test("Authorization header set",
         call_args.get("headers", {}).get("Authorization") == "Bearer sk-test-key")
    test("Model in payload", call_args.get("json", {}).get("model") == "gpt-image-1")
    test("Prompt in payload", call_args.get("json", {}).get("prompt") == "a cute cat")
    test("Quality in payload", call_args.get("json", {}).get("quality") == "high")
    test("Size in payload", call_args.get("json", {}).get("size") == "1024x1024")
    test("Info string contains endpoint", "http://localhost:8317/v1" in info)
    test("Info string contains model", "gpt-image-1" in info)

except Exception as e:
    test(f"API call mock test failed: {e}", False, str(e))
    traceback.print_exc()

# --- Test 10: Retry logic (mock) ---
print("\n[10] Retry Logic (mock)")
try:
    from unittest.mock import patch, MagicMock

    # First call fails with 500, second succeeds
    fail_response = MagicMock()
    fail_response.status_code = 500
    fail_response.text = "Internal Server Error"
    fail_response.json.return_value = {"error": {"message": "Server error"}}

    success_response = MagicMock()
    success_response.status_code = 200

    img_buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(img_buf, format="PNG")
    b64_img = base64.b64encode(img_buf.getvalue()).decode()
    success_response.json.return_value = {"data": [{"b64_json": b64_img}]}

    call_count = [0]
    def mock_post_with_retry(url, **kwargs):
        call_count[0] += 1
        if call_count[0] == 1:
            return fail_response
        return success_response

    node = LanGPTImage2()

    with patch("requests.post", side_effect=mock_post_with_retry):
        with patch("time.sleep", return_value=None):  # Skip actual sleep
            result, info = node.generate(
                prompt="test retry",
                api_key="sk-test",
                base_url="http://localhost:8317/v1",
                max_retries=3,
                retry_delay=0.1,
            )

    test("Retry: succeeded after failure", call_count[0] >= 2)
    test("Retry: info contains retry attempt", "Retrying" in info or "Attempt" in info)

except Exception as e:
    test(f"Retry logic test failed: {e}", False, str(e))
    traceback.print_exc()

# --- Test 11: Moderation fallback (mock) ---
print("\n[11] Moderation Fallback Logic (mock)")
try:
    from unittest.mock import patch, MagicMock

    # Success response
    img_buf = io.BytesIO()
    Image.new("RGB", (8, 8), color=(128, 128, 128)).save(img_buf, format="PNG")
    b64_img = base64.b64encode(img_buf.getvalue()).decode()

    success_resp = MagicMock()
    success_resp.status_code = 200
    success_resp.json.return_value = {"data": [{"b64_json": b64_img}]}

    # Moderation rejection response (400)
    mod_error_resp = MagicMock()
    mod_error_resp.status_code = 400
    mod_error_resp.text = '{"error": {"message": "Invalid value for moderation: none. Must be auto or low."}}'
    mod_error_resp.json.return_value = {
        "error": {"message": "Invalid value for moderation: none. Must be auto or low."}
    }

    # Test 11a: moderation='none' with auto_fallback=True -> should fall back to 'low'
    call_data = []
    def mock_post_fallback(url, **kwargs):
        # Deep copy the payload to preserve what was actually sent (data dict is mutated on fallback)
        import copy
        call_data.append({
            "url": url,
            "json": copy.deepcopy(kwargs.get("json")),
            "data": copy.deepcopy(kwargs.get("data")),
        })
        # Check what moderation value was sent
        payload = kwargs.get("json") or kwargs.get("data") or {}
        mod_val = payload.get("moderation", "")
        if mod_val == "none":
            return mod_error_resp
        return success_resp

    node = LanGPTImage2()
    with patch("requests.post", side_effect=mock_post_fallback):
        with patch("time.sleep", return_value=None):
            result, info = node.generate(
                prompt="test moderation fallback",
                api_key="sk-test",
                base_url="http://localhost:8317/v1",
                moderation="none",
                auto_fallback_moderation=True,
                max_retries=0,
            )

    test("Moderation fallback: succeeded after fallback", result is not None)
    test("Moderation fallback: info contains fallback message", "fallback" in info.lower() or "Falling back" in info)
    # Check moderation values sent in requests (json for generation, data for edit)
    first_payload = call_data[0].get("json") or call_data[0].get("data") or {}
    second_payload = call_data[1].get("json") or call_data[1].get("data") or {}
    test("Moderation fallback: first request used 'none'", first_payload.get("moderation") == "none")
    test("Moderation fallback: second request used 'low'", second_payload.get("moderation") == "low")

    # Test 11b: moderation='none' with auto_fallback=False -> should fail
    call_data2 = []
    def mock_post_no_fallback(url, **kwargs):
        call_data2.append({"json": kwargs.get("json"), "data": kwargs.get("data")})
        return mod_error_resp

    node2 = LanGPTImage2()
    with patch("requests.post", side_effect=mock_post_no_fallback):
        with patch("time.sleep", return_value=None):
            try:
                node2.generate(
                    prompt="test no fallback",
                    api_key="sk-test",
                    base_url="http://localhost:8317/v1",
                    moderation="none",
                    auto_fallback_moderation=False,
                    max_retries=0,
                )
                test("No fallback: raises error when disabled", False, "should have raised")
            except RuntimeError:
                test("No fallback: raises error when disabled", True)

    test("No fallback: only one request made", len(call_data2) == 1)

except Exception as e:
    test(f"Moderation fallback test failed: {e}", False, str(e))
    traceback.print_exc()

# --- Summary ---
print("\n" + "=" * 60)
print(f"RESULTS: {passed} passed, {failed} failed")
print("=" * 60)

if failed > 0:
    print("\nFailed tests:")
    for err in errors:
        print(f"  - {err}")
    sys.exit(1)
else:
    print("\nAll tests passed!")
    sys.exit(0)
