#!/usr/bin/env python3
import sys
from pathlib import Path


HELPER = r'''

# ---- reasoning helpers ----

REASONING_START_TAGS = ("<think>", "<thinking>")
REASONING_END_TAGS = ("</think>", "</thinking>")


def _expects_reasoning(model_name: str, chat_template_kwargs: Optional[dict]) -> bool:
    name = (model_name or "").lower()
    if isinstance(chat_template_kwargs, dict):
        enabled = chat_template_kwargs.get("enable_thinking")
        if enabled is not None:
            return bool(enabled)
    return any(
        marker in name
        for marker in (
            "deepseek-r1",
            "deepseek-v3",
            "qwen3",
            "qwen3.5",
            "qwen3.6",
            "gemma-4",
            "gemma4",
            "gpt-oss",
            "reason",
            "thinking",
        )
    )


class ReasoningSplitter:
    def __init__(self, infer_missing_start: bool):
        self.probing = infer_missing_start
        self.probe = ""
        self.in_reasoning = False
        self.tag = ""

    def process(self, text: str) -> List[Dict[str, str]]:
        if not text:
            return []

        if self.probing:
            self.probe += text
            lower = self.probe.lower()

            open_at, open_tag = self._find_any(lower, REASONING_START_TAGS)
            if open_at >= 0:
                before = self.probe[:open_at]
                after = self.probe[open_at + len(open_tag):]
                self.probe = ""
                self.probing = False
                self.in_reasoning = True
                parts = []
                if before:
                    parts.append({"content": before})
                parts.extend(self._process_tagged(after))
                return parts

            close_at, close_tag = self._find_any(lower, REASONING_END_TAGS)
            if close_at >= 0:
                reasoning = self.probe[:close_at]
                after = self.probe[close_at + len(close_tag):]
                self.probe = ""
                self.probing = False
                self.in_reasoning = False
                parts = []
                if reasoning:
                    parts.append({"reasoning_content": reasoning})
                parts.extend(self._process_tagged(after))
                return parts

            if len(self.probe) < 8192:
                return []

            text = self.probe
            self.probe = ""
            self.probing = False

        return self._process_tagged(text)

    def flush(self) -> List[Dict[str, str]]:
        parts = []
        if self.probe:
            parts.append({"content": self.probe})
            self.probe = ""
            self.probing = False
        if self.tag:
            key = "reasoning_content" if self.in_reasoning else "content"
            parts.append({key: self.tag})
            self.tag = ""
        return parts

    def _process_tagged(self, text: str) -> List[Dict[str, str]]:
        parts: List[Dict[str, str]] = []
        for char in text:
            if self.tag:
                self.tag += char
                if ">" in self.tag:
                    tag = self.tag.lower()
                    if any(tag.startswith(start[:-1]) for start in REASONING_START_TAGS):
                        self.in_reasoning = True
                    elif any(tag.startswith(end[:-1]) for end in REASONING_END_TAGS):
                        self.in_reasoning = False
                    else:
                        self._append(parts, self.tag)
                    self.tag = ""
                    continue
                if self._is_reasoning_tag_prefix(self.tag):
                    continue
                self._append(parts, self.tag)
                self.tag = ""
                continue

            if char == "<":
                self.tag = "<"
                continue

            self._append(parts, char)
        return parts

    def _append(self, parts: List[Dict[str, str]], text: str) -> None:
        key = "reasoning_content" if self.in_reasoning else "content"
        if parts and key in parts[-1]:
            parts[-1][key] += text
        else:
            parts.append({key: text})

    def _is_reasoning_tag_prefix(self, tag: str) -> bool:
        lower = tag.lower()
        return any(start.startswith(lower) for start in REASONING_START_TAGS) or any(
            end.startswith(lower) for end in REASONING_END_TAGS
        )

    def _find_any(self, text: str, needles: tuple[str, ...]) -> tuple[int, str]:
        best_at = -1
        best_needle = ""
        for needle in needles:
            at = text.find(needle)
            if at >= 0 and (best_at < 0 or at < best_at):
                best_at = at
                best_needle = needle
        return best_at, best_needle


def split_reasoning_text(text: str, infer_missing_start: bool) -> tuple[str, str]:
    splitter = ReasoningSplitter(infer_missing_start)
    reasoning = []
    content = []
    for part in splitter.process(text) + splitter.flush():
        if part.get("reasoning_content"):
            reasoning.append(part["reasoning_content"])
        if part.get("content"):
            content.append(part["content"])
    return "".join(reasoning), "".join(content)
'''


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: openarc-reasoning-content.py /path/to/openai.py")

    path = Path(sys.argv[1])
    source = path.read_text()

    if "class ReasoningSplitter:" not in source:
        source = source.replace(
            "\n\n# ---- endpoints ----\n",
            HELPER + "\n\n# ---- endpoints ----\n",
        )

    source = source.replace(
        "                cancel_request_id = None\n\n                try:",
        "                cancel_request_id = None\n"
        "                reasoning_splitter = ReasoningSplitter(\n"
        "                    _expects_reasoning(model_name, request.chat_template_kwargs)\n"
        "                )\n\n"
        "                try:",
        1,
    )

    old_chunk = '''                        elif not tool_calls and not tool_call_started:
                            chunk_payload = {
                                "id": request_id,
                                "object": "chat.completion.chunk",
                                "created": created_ts,
                                "model": model_name,
                                "choices": [
                                    {
                                        "index": 0,
                                        "delta": {"content": item},
                                        "finish_reason": None,
                                    }
                                ],
                            }
                            yield (f"data: {json.dumps(chunk_payload)}\\n\\n").encode()
'''
    new_chunk = '''                        elif not tool_calls and not tool_call_started:
                            for delta in reasoning_splitter.process(item):
                                chunk_payload = {
                                    "id": request_id,
                                    "object": "chat.completion.chunk",
                                    "created": created_ts,
                                    "model": model_name,
                                    "choices": [
                                        {
                                            "index": 0,
                                            "delta": delta,
                                            "finish_reason": None,
                                        }
                                    ],
                                }
                                yield (f"data: {json.dumps(chunk_payload)}\\n\\n").encode()
'''
    source = source.replace(old_chunk, new_chunk, 1)

    old_final = '''                prompt_tokens = (metrics_data or {}).get("input_token", 0)
'''
    new_final = '''                for delta in reasoning_splitter.flush():
                    chunk_payload = {
                        "id": request_id,
                        "object": "chat.completion.chunk",
                        "created": created_ts,
                        "model": model_name,
                        "choices": [
                            {
                                "index": 0,
                                "delta": delta,
                                "finish_reason": None,
                            }
                        ],
                    }
                    yield (f"data: {json.dumps(chunk_payload)}\\n\\n").encode()

                prompt_tokens = (metrics_data or {}).get("input_token", 0)
'''
    source = source.replace(old_final, new_final, 1)

    source = source.replace(
        "            tool_calls = parse_tool_calls(text)\n"
        "            message = {\"role\": \"assistant\"}\n",
        "            reasoning_content, response_text = split_reasoning_text(\n"
        "                text, _expects_reasoning(model_name, request.chat_template_kwargs)\n"
        "            )\n"
        "            tool_calls = parse_tool_calls(response_text)\n"
        "            message = {\"role\": \"assistant\"}\n",
        1,
    )
    source = source.replace(
        "            else:\n"
        "                message[\"content\"] = text\n\n"
        "            return {\n",
        "            else:\n"
        "                message[\"content\"] = response_text\n"
        "            if reasoning_content:\n"
        "                message[\"reasoning_content\"] = reasoning_content\n\n"
        "            return {\n",
        1,
    )

    path.write_text(source)


if __name__ == "__main__":
    main()
