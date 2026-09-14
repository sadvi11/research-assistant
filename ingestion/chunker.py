"""Structure-aware chunking.

Two rules drive the design:

1. A chunk that loses its context is worse than no chunk. Headings are carried
   into every chunk beneath them as a `section_path`, so a chunk reading
   "revenue grew 3%" still knows which document and section it came from.
2. Never split mid-sentence when a sentence boundary is available. A retrieved
   half-sentence answers half a question, which is the failure mode that
   produces confident wrong answers.

Token counts here are an approximation (see `approx_tokens`). Chunk boundaries
do not need to be exact; the context-window budget check before generation is
the place where accuracy matters, and that uses the API's own counter when
credentials are available.
"""
from __future__ import annotations

import hashlib
import re
from dataclasses import dataclass

from schemas import Chunk, DocumentMetadata

# Claude averages ~3.5 characters per token for English prose.
_CHARS_PER_TOKEN = 3.5

_HEADING = re.compile(r"^(#{1,6})\s+(.+?)\s*$", re.MULTILINE)
_SENTENCE_END = re.compile(r"(?<=[.!?])\s+(?=[A-Z(\[\"'])")
_PARAGRAPH = re.compile(r"\n\s*\n")


def approx_tokens(text: str) -> int:
    """Approximate token count. Deliberately cheap - no model call."""
    return max(1, int(len(text) / _CHARS_PER_TOKEN))


@dataclass(frozen=True)
class _Block:
    """A structural unit of the document: one paragraph under a heading path."""

    text: str
    start: int
    end: int
    section_path: tuple[str, ...]


class Chunker:
    """Splits documents into overlapping, context-preserving chunks."""

    def __init__(self, *, chunk_size_tokens: int = 400, overlap_tokens: int = 60) -> None:
        if overlap_tokens >= chunk_size_tokens:
            raise ValueError("overlap must be smaller than chunk size")
        if chunk_size_tokens <= 0:
            raise ValueError("chunk size must be positive")
        self.chunk_size = chunk_size_tokens
        self.overlap = overlap_tokens

    # -- structure ---------------------------------------------------------

    def _blocks(self, text: str) -> list[_Block]:
        """Split into paragraphs, tracking the heading hierarchy above each."""
        headings: list[tuple[int, int, str]] = [
            (m.start(), len(m.group(1)), m.group(2).strip()) for m in _HEADING.finditer(text)
        ]

        def path_at(pos: int) -> tuple[str, ...]:
            stack: list[tuple[int, str]] = []
            for h_pos, level, title in headings:
                if h_pos > pos:
                    break
                while stack and stack[-1][0] >= level:
                    stack.pop()
                stack.append((level, title))
            return tuple(title for _, title in stack)

        blocks: list[_Block] = []
        cursor = 0
        for para in _PARAGRAPH.split(text):
            if not para.strip():
                cursor += len(para) + 2
                continue
            start = text.find(para, cursor)
            if start == -1:
                start = cursor
            end = start + len(para)
            cursor = end
            # A lone heading line is context for what follows, not content itself.
            stripped = para.strip()
            if _HEADING.fullmatch(stripped):
                continue
            blocks.append(_Block(stripped, start, end, path_at(start)))
        return blocks

    @staticmethod
    def _sentences(text: str) -> list[str]:
        parts = [s.strip() for s in _SENTENCE_END.split(text) if s.strip()]
        return parts or ([text.strip()] if text.strip() else [])

    # -- chunking ----------------------------------------------------------

    def split(self, text: str, metadata: DocumentMetadata) -> list[Chunk]:
        """Split `text` into chunks, each carrying full source metadata.

        Invariant: every block of the source appears in at least one chunk.
        Overlap is applied *within* a section only - carrying the tail of one
        section into the next would mislabel it and blur the boundary.
        """
        if not text or not text.strip():
            return []

        blocks = self._blocks(text)
        if not blocks:
            return []

        chunks: list[Chunk] = []
        for section_path, section_blocks in self._group_by_section(blocks):
            for body, start, end in self._pack_section(section_blocks):
                chunks.append(
                    self._make_chunk(body, start, end, section_path, len(chunks), metadata)
                )
        return chunks

    @staticmethod
    def _group_by_section(blocks: list[_Block]) -> list[tuple[tuple[str, ...], list[_Block]]]:
        """Consecutive blocks sharing a heading path, in document order."""
        groups: list[tuple[tuple[str, ...], list[_Block]]] = []
        for block in blocks:
            if groups and groups[-1][0] == block.section_path:
                groups[-1][1].append(block)
            else:
                groups.append((block.section_path, [block]))
        return groups

    def _pack_section(self, blocks: list[_Block]) -> list[tuple[str, int, int]]:
        """Pack one section's blocks into overlapping chunks."""
        out: list[tuple[str, int, int]] = []
        buf: list[str] = []
        buf_tokens = 0
        buf_start: int | None = None
        buf_end = 0

        def emit() -> None:
            nonlocal buf, buf_tokens, buf_start, buf_end
            if not buf or buf_start is None:
                return
            body = "\n\n".join(buf).strip()
            if body:
                out.append((body, buf_start, buf_end))
            carry, carry_tokens = self._overlap_tail(body)
            if carry:
                buf, buf_tokens = [carry], carry_tokens
                buf_start = max(buf_end - len(carry), 0)
            else:
                buf, buf_tokens, buf_start = [], 0, None

        for block in blocks:
            block_tokens = approx_tokens(block.text)

            if block_tokens > self.chunk_size:
                # Oversized block: flush what we have, then split it on sentences.
                if buf and buf_start is not None:
                    body = "\n\n".join(buf).strip()
                    if body:
                        out.append((body, buf_start, buf_end))
                buf, buf_tokens, buf_start = [], 0, None
                out.extend(self._split_oversized(block))
                continue

            if buf and buf_tokens + block_tokens > self.chunk_size:
                emit()

            if buf_start is None:
                buf_start = block.start
            buf.append(block.text)
            buf_tokens += block_tokens
            buf_end = block.end

        # Final buffer. Emit unless it is purely overlap already covered above.
        if buf and buf_start is not None:
            body = "\n\n".join(buf).strip()
            if body and not (out and body in out[-1][0]):
                out.append((body, buf_start, buf_end))
        return out

    def _split_oversized(self, block: _Block) -> list[tuple[str, int, int]]:
        """Sentence-wise split of a block that exceeds the chunk budget."""
        out: list[tuple[str, int, int]] = []
        current: list[str] = []
        current_tokens = 0
        offset = block.start
        run_start = block.start

        for sentence in self._sentences(block.text):
            s_tokens = approx_tokens(sentence)
            if current and current_tokens + s_tokens > self.chunk_size:
                body = " ".join(current)
                out.append((body, run_start, run_start + len(body)))
                carry, current_tokens = self._overlap_tail(body)
                current = [carry] if carry else []
                run_start = offset - len(carry) if carry else offset
            current.append(sentence)
            current_tokens += s_tokens
            offset += len(sentence) + 1

        if current:
            body = " ".join(current)
            out.append((body, run_start, min(run_start + len(body), block.end)))
        return out

    def _overlap_tail(self, text: str) -> tuple[str, int]:
        """Last whole sentences of `text` totalling roughly `self.overlap` tokens."""
        if self.overlap <= 0:
            return "", 0
        sentences = self._sentences(text)
        tail: list[str] = []
        tokens = 0
        for sentence in reversed(sentences):
            s_tokens = approx_tokens(sentence)
            if tokens + s_tokens > self.overlap and tail:
                break
            tail.insert(0, sentence)
            tokens += s_tokens
        return (" ".join(tail), tokens) if tail else ("", 0)

    @staticmethod
    def _make_chunk(
        text: str,
        start: int,
        end: int,
        section_path: tuple[str, ...],
        ordinal: int,
        metadata: DocumentMetadata,
    ) -> Chunk:
        digest = hashlib.sha256(f"{metadata.document_id}:{ordinal}:{text}".encode()).hexdigest()
        return Chunk(
            chunk_id=f"{metadata.document_id}::{ordinal:04d}::{digest[:12]}",
            document_id=metadata.document_id,
            text=text,
            ordinal=ordinal,
            char_start=start,
            char_end=end,
            metadata=metadata,
            section_path=section_path,
        )
