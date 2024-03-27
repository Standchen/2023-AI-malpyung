import re

from soynlp.normalizer import emoticon_normalize


def _normalize(s):
    if s is None:
        return None
    """Perform normalization."""
    # (1) Remove unrecognized substrings.
    pattern = r"&(?:name\d|account\d|affiliation\d|others)&"
    s = re.sub(pattern, "", s)
    # (2) Shorten repeated punctuations.
    s = re.sub(r"([*-.\?#@+,<>%~`!$^&\(\):;])\1+", r"\1\1", s)
    # (3) Shorten repeated emoticons.
    s = emoticon_normalize(s, num_repeats=2)
    return s


def bert_preprocess(example, tokenizer, labels):
    # Fetch the batch.
    text = example["input"]["form"]
    target = example["input"]["target"]["form"]

    # Normalize.
    text = _normalize(text)
    target = _normalize(target)

    # Tokenize.
    res = tokenizer(text, target, truncation=True)
    # Tailor the labels.
    res["labels"] = [1.0 if example["output"][k] == "True" else 0.0 for k in labels]
    return res

def gpt_preprocess(examples, tokenizer, labels_to_korean, mode: str):
    assert mode in ("train", "val")
    instruction = "입력으로 주어지는 텍스트에서 대상에 대하여 나타나는 화자의 감정 상태를 파악하세요. 감정 상태는 기쁨, 기대, 신뢰, 깜짝, 혐오, 공포, 분노, 슬픔으로 이루어져 있습니다."
    inputs = [_normalize(text) for text in examples["text"]]
    targets = [_normalize(targets) for targets in examples["target"]]

    queries = []
    if mode == "train":
        for inp, tar, out in zip(inputs, targets, examples["output"]):
            for k, v in labels_to_korean.items():
                if out[k] == "True":
                    queries.append(f"### 명령어: {instruction}\n\n### 입력: {inp}\n\n### 대상: {tar}\n\n### 응답: {v}")
        return tokenizer(queries, return_token_type_ids=True)
    elif mode == "val":
        for inp, tar in zip(inputs, targets):
            queries.append(f"### 명령어: {instruction}\n\n### 입력: {inp}\n\n### 대상: {tar}\n\n### 응답:")
        return tokenizer(queries, return_token_type_ids=False)
