#!/usr/bin/env python3
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: openarc-tokenizer-fallback.py /path/to/vlm.py")

    path = Path(sys.argv[1])
    source = path.read_text()

    source = source.replace(
        "import gc\n\nimport logging",
        "import gc\nimport json\n\nimport logging",
    )
    source = source.replace(
        "from io import BytesIO\n",
        "from io import BytesIO\nfrom pathlib import Path\n",
    )

    helper = '''

def load_vlm_tokenizer(model_path: str):
    """
    Some OpenVINO VLM exports save tokenizer_config.json with
    tokenizer_class=TokenizersBackend for the OpenVINO tokenizer assets. OpenArc
    still needs a Transformers tokenizer here only to render the chat template,
    so fall back to the source tokenizer referenced by openvino_config.json.
    """
    try:
        return AutoTokenizer.from_pretrained(model_path)
    except ValueError as exc:
        if "Tokenizer class TokenizersBackend" not in str(exc):
            raise

    config_path = Path(model_path) / "openvino_config.json"
    if not config_path.exists():
        raise

    with config_path.open("r", encoding="utf-8") as handle:
        ov_config = json.load(handle)

    quant_config = ov_config.get("quantization_config") or {}
    tokenizer_repo = quant_config.get("tokenizer")
    if not tokenizer_repo:
        for section in (quant_config.get("quantization_configs") or {}).values():
            if isinstance(section, dict) and section.get("tokenizer"):
                tokenizer_repo = section["tokenizer"]
                break

    if not tokenizer_repo:
        raise

    logger.warning(
        "Falling back to source tokenizer %s for OpenVINO model at %s",
        tokenizer_repo,
        model_path,
    )
    return AutoTokenizer.from_pretrained(tokenizer_repo)
'''

    source = source.replace(
        "logger.setLevel(logging.INFO)\n\n\nclass OVGenAI_VLM:",
        "logger.setLevel(logging.INFO)" + helper + "\n\nclass OVGenAI_VLM:",
    )
    source = source.replace(
        "self.tokenizer = AutoTokenizer.from_pretrained(loader.model_path)",
        "self.tokenizer = load_vlm_tokenizer(loader.model_path)",
    )

    path.write_text(source)


if __name__ == "__main__":
    main()
