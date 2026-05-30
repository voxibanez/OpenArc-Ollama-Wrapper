#!/usr/bin/env python3
import sys
from pathlib import Path


def main() -> None:
    if len(sys.argv) != 2:
        raise SystemExit("usage: openarc-streaming-unicode.py /path/to/streamers.py")

    path = Path(sys.argv[1])
    source = path.read_text()

    old_write = '''        # Only emit when we've reached the chunk boundary
        if self.since_last_emit >= self.tokens_len:
            text = self.decoder_tokenizer.decode(self.tokens_cache)
            # Emit only the newly materialized portion
            if len(text) > self.last_print_len:
                chunk = text[self.last_print_len:]
                if chunk:
                    self.text_queue.put_nowait(chunk)
                self.last_print_len = len(text)
            self.since_last_emit = 0
'''
    new_write = '''        # Only emit when we've reached the chunk boundary
        if self.since_last_emit >= self.tokens_len:
            self._emit_decoded(final=False)
            self.since_last_emit = 0
'''
    if old_write not in source:
        raise SystemExit("write emission block not found")
    source = source.replace(old_write, new_write, 1)

    helper = '''    def _emit_decoded(self, final: bool = False) -> None:
        text = self.decoder_tokenizer.decode(self.tokens_cache)
        if len(text) <= self.last_print_len:
            return

        chunk = text[self.last_print_len:]

        # Tokenizers with byte-fallback can temporarily decode an incomplete
        # multi-byte character as U+FFFD. Do not stream that unstable suffix;
        # wait for later tokens to resolve it. At EOS, drop any replacement
        # characters that never stabilized.
        replacement_at = chunk.find("\\ufffd")
        if replacement_at >= 0:
            if final:
                chunk = chunk.replace("\\ufffd", "")
                if chunk:
                    self.text_queue.put_nowait(chunk)
                self.last_print_len = len(text)
            else:
                stable = chunk[:replacement_at]
                if stable:
                    self.text_queue.put_nowait(stable)
                    self.last_print_len += len(stable)
            return

        if chunk:
            self.text_queue.put_nowait(chunk)
        self.last_print_len = len(text)

'''
    if "    def _emit_decoded(" not in source:
        source = source.replace("    def cancel(self) -> None:\n", helper + "    def cancel(self) -> None:\n", 1)

    old_end = '''    def end(self) -> None:
        # Flush any remaining tokens at the end
        text = self.decoder_tokenizer.decode(self.tokens_cache)
        if len(text) > self.last_print_len:
            chunk = text[self.last_print_len:]
            if chunk:
                self.text_queue.put_nowait(chunk)
        # Signal completion
        self.text_queue.put_nowait(None)'''
    new_end = '''    def end(self) -> None:
        # Flush any remaining tokens at the end
        self._emit_decoded(final=True)
        # Signal completion
        self.text_queue.put_nowait(None)'''
    if old_end not in source:
        raise SystemExit("end emission block not found")
    source = source.replace(old_end, new_end, 1)

    path.write_text(source)


if __name__ == "__main__":
    main()
