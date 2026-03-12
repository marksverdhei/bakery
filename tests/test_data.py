from bakery.data import PromptBakingDataset, prompt_baking_collator


def test_dataset_creation():
    prompts = ["hello", "world"]
    ds = PromptBakingDataset(prompts)
    assert len(ds) == 2
    assert ds[0]["user_messages"] == "hello"
    assert "responses" not in ds.column_names


def test_dataset_with_responses():
    prompts = ["hello", "world"]
    responses = ["hi there", "earth"]
    ds = PromptBakingDataset(prompts, responses)
    assert len(ds) == 2
    assert ds[0]["responses"] == "hi there"


def test_collator():
    features = [
        {"user_messages": "hello", "responses": "hi"},
        {"user_messages": "world", "responses": "earth"},
    ]
    batch = prompt_baking_collator(features)
    assert batch["user_messages"] == ["hello", "world"]
    assert batch["responses"] == ["hi", "earth"]


def test_collator_handles_lists():
    features = [
        {"user_messages": ["a", "b"], "responses": ["c", "d"]},
    ]
    batch = prompt_baking_collator(features)
    assert batch["user_messages"] == ["a", "b"]
    assert batch["responses"] == ["c", "d"]
