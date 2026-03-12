from transformers import HfArgumentParser
from bakery.config import BakeryConfig, DataConfig, LoraConfig


def test_parser_accepts_flat_args():
    parser = HfArgumentParser((BakeryConfig, DataConfig, LoraConfig))
    baking, data, lora = parser.parse_args_into_dataclasses(args=[
        "--output_dir", "/tmp/test",
        "--system_prompt", "You are helpful.",
        "--model_name_or_path", "gpt2",
        "--r", "16",
    ])
    assert baking.system_prompt == "You are helpful."
    assert data.model_name_or_path == "gpt2"
    assert lora.r == 16
    assert baking.remove_unused_columns is False


def test_default_lora_targets():
    lora = LoraConfig()
    assert "q_proj" in lora.target_modules
    assert len(lora.target_modules) == 7
