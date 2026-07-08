"""
Build chat messages for Dolly-style supervised fine-tuning.
"""


def build_user_content(sample: dict) -> str:
    instruction = sample["instruction"].strip()
    context = sample.get("context", "").strip()

    if not context:
        return instruction

    return (
        f"{instruction}\n\n"
        "Context:\n"
        f"{context}"
    )


def build_messages(sample: dict) -> list[dict[str, str]]:
    return [
        {
            "role": "user",
            "content": build_user_content(sample),
        },
        {
            "role": "assistant",
            "content": sample["response"].strip(),
        },
    ]
