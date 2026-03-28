#!/usr/bin/env python3
"""
audiobook_maker.py
==================
Converts an EPUB or PDF into a chapter-structured audiobook using Kokoro TTS.

Kokoro produces expressive, natural-sounding narration with proper prosody
and emotional inflection — far beyond flat text-to-speech.

Always outputs:
  • Per-chapter WAV files
  • A combined .m4b file with chapter markers (requires ffmpeg in PATH)

Always uses CUDA (GPU) for synthesis. Make sure your PyTorch install has
CUDA support: https://pytorch.org/get-started/locally/

Install dependencies:
    pip install kokoro soundfile numpy ebooklib beautifulsoup4 PyMuPDF

Required for M4B export:
    Install ffmpeg and add to PATH: https://ffmpeg.org/download.html

Usage:
    python audiobook_maker.py my_book.epub
    python audiobook_maker.py my_book.pdf --voice am_michael --speed 0.95
    python audiobook_maker.py my_book.epub --list-voices

Available voices (American English — best for audiobooks):
    af_heart    ← warm, expressive female  [DEFAULT, recommended]
    af_bella    ← bright, clear female
    af_nicole   ← calm, measured female
    am_adam     ← warm, deep male
    am_michael  ← clear, authoritative male

Voice naming convention:  a=American, b=British | f=female, m=male
British English voices:   bf_emma, bf_isabella, bm_george, bm_lewis
"""

import warnings
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

import os
import re
import sys
import json
import shutil
import argparse
import tempfile
from pathlib import Path

import numpy as np
import soundfile as sf




# ── Text cleaning ─────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Normalize text for natural TTS output."""
    # Collapse whitespace and newlines
    text = re.sub(r'[ \t]+', ' ', text)
    text = re.sub(r'\n{3,}', '\n\n', text)
    text = re.sub(r'\n', ' ', text)

    # Fix ligatures (common in PDFs)
    text = text.replace('ﬁ', 'fi').replace('ﬂ', 'fl').replace('ﬀ', 'ff')
    text = text.replace('ﬃ', 'ffi').replace('ﬄ', 'ffl')

    # Fix missing space after sentence-ending punctuation
    text = re.sub(r'([.!?])([A-Z])', r'\1 \2', text)

    # Expand common abbreviations so TTS reads them naturally
    abbrevs = {
        r'\bMr\.': 'Mister', r'\bMrs\.': 'Missus', r'\bMs\.': 'Miss',
        r'\bDr\.': 'Doctor', r'\bProf\.': 'Professor', r'\bSt\.': 'Saint',
        r'\bvs\.': 'versus', r'\betc\.': 'et cetera', r'\bi\.e\.': 'that is',
        r'\be\.g\.': 'for example', r'\bU\.S\.': 'U.S.', r'\bU\.K\.': 'U.K.',
    }
    for pattern, replacement in abbrevs.items():
        text = re.sub(pattern, replacement, text)

    # Convert em-dashes/en-dashes to natural pauses
    text = re.sub(r'\s*[—–]\s*', ' — ', text)

    # Remove stray page numbers (isolated digits on their own)
    text = re.sub(r'(?<!\w)\d{1,4}(?!\w)', '', text)

    # Clean up multiple spaces created above
    text = re.sub(r' {2,}', ' ', text)

    return text.strip()


# ── Text chunking ─────────────────────────────────────────────────────────────

def chunk_text(text: str, max_chars: int = 400) -> list[str]:
    """
    Split text into TTS-friendly chunks at natural sentence boundaries.
    Kokoro works best with chunks of ~1–3 sentences rather than huge blocks.
    Splits at: sentence endings → commas → words (last resort).
    """
    # Split into sentences
    sentences = re.split(r'(?<=[.!?…])\s+', text)
    chunks = []
    current = ""

    for sentence in sentences:
        sentence = sentence.strip()
        if not sentence:
            continue

        if not current:
            current = sentence
        elif len(current) + 1 + len(sentence) <= max_chars:
            current += " " + sentence
        else:
            chunks.append(current)
            # If a single sentence is still too long, split on commas
            if len(sentence) > max_chars:
                parts = re.split(r'(?<=,)\s+', sentence)
                sub = ""
                for part in parts:
                    if not sub:
                        sub = part
                    elif len(sub) + 1 + len(part) <= max_chars:
                        sub += " " + part
                    else:
                        chunks.append(sub)
                        sub = part
                if sub:
                    chunks.append(sub)
                current = ""
            else:
                current = sentence

    if current:
        chunks.append(current)

    return [c for c in chunks if c.strip()]


# ── EPUB extraction ───────────────────────────────────────────────────────────

def extract_epub(path: Path) -> list[tuple[str, str]]:
    """
    Extract chapters from an EPUB file.
    Returns list of (title, text) tuples in reading order.
    Respects the spine order and uses headings as chapter titles.
    """
    import ebooklib
    from ebooklib import epub
    from bs4 import BeautifulSoup

    print("  Reading EPUB structure...")
    book = epub.read_epub(str(path), options={'ignore_ncx': False})

    # Build a title map from the TOC for better chapter naming
    toc_titles = {}
    def walk_toc(items):
        for item in items:
            if isinstance(item, epub.Link):
                # href may have fragment: e.g. "chapter01.xhtml#ch1"
                href = item.href.split('#')[0]
                toc_titles[href] = item.title
            elif isinstance(item, tuple):
                walk_toc(item[1])
    walk_toc(book.toc)

    chapters = []
    chapter_num = 0

    for item in book.get_items_of_type(ebooklib.ITEM_DOCUMENT):
        href = item.get_name()
        soup = BeautifulSoup(item.get_content(), 'html.parser')

        # Remove non-content elements
        for tag in soup(['script', 'style', 'nav', 'aside', 'figure', 'figcaption']):
            tag.decompose()

        # Determine chapter title
        title = toc_titles.get(href, None)
        if not title:
            heading = soup.find(['h1', 'h2', 'h3'])
            title = heading.get_text(strip=True) if heading else None

        # Extract text
        text = soup.get_text(separator=' ')
        text = clean_text(text)

        # Skip nav pages, copyright pages, very short content
        if len(text.split()) < 100:
            continue

        chapter_num += 1
        if not title:
            title = f"Chapter {chapter_num}"

        chapters.append((title, text))

    return chapters


# ── PDF extraction ────────────────────────────────────────────────────────────

def extract_pdf(path: Path) -> list[tuple[str, str]]:
    """
    Extract chapters from a PDF file.
    Uses the PDF's built-in TOC when available; falls back to heading
    pattern detection, then to fixed-size chunking as a last resort.
    """
    import fitz  # PyMuPDF

    print("  Reading PDF structure...")
    doc = fitz.open(str(path))
    toc = doc.get_toc()  # [[level, title, page_number], ...]

    # ── Strategy 1: Use embedded TOC ─────────────────────────────────────────
    if toc:
        print(f"  Found TOC with {len(toc)} entries — using embedded chapter structure")
        # Filter to top-level entries (level 1) as chapter boundaries
        top_level = [(title.strip(), page - 1) for level, title, page in toc if level == 1]

        chapters = []
        for i, (title, start_page) in enumerate(top_level):
            end_page = top_level[i + 1][1] if i + 1 < len(top_level) else len(doc)
            text = ""
            for p in range(start_page, min(end_page, len(doc))):
                text += doc[p].get_text()
            text = clean_text(text)
            if len(text.split()) >= 100:
                chapters.append((title, text))

        if chapters:
            return chapters

    # ── Strategy 2: Detect "Chapter N" headings ───────────────────────────────
    print("  No TOC found — scanning for chapter headings")
    full_text = "\n".join(page.get_text() for page in doc)

    # Match patterns like "Chapter 1", "Chapter One", "CHAPTER I", etc.
    pattern = r'((?:Chapter|CHAPTER)\s+(?:\d+|[A-Z][a-z]+|[IVXLCDM]+)(?:\s*[:\-—]\s*[^\n]+)?)'
    parts = re.split(pattern, full_text)

    if len(parts) >= 3:
        chapters = []
        # parts[0] is pre-chapter content (intro/TOC), skip or include as Preface
        preamble = clean_text(parts[0])
        if len(preamble.split()) >= 150:
            chapters.append(("Preface", preamble))

        for i in range(1, len(parts) - 1, 2):
            title = parts[i].strip()
            body = parts[i + 1] if i + 1 < len(parts) else ""
            text = clean_text(body)
            if len(text.split()) >= 100:
                chapters.append((title, text))

        if chapters:
            return chapters

    # ── Strategy 3: Fixed-size chunks ────────────────────────────────────────
    print("  No headings found — splitting into ~3,000-word parts")
    full_text = clean_text(full_text)
    words = full_text.split()
    chunk_size = 3000
    chapters = []
    for i in range(0, len(words), chunk_size):
        chunk = " ".join(words[i:i + chunk_size])
        chapters.append((f"Part {i // chunk_size + 1}", chunk))

    return chapters


# ── TTS synthesis ─────────────────────────────────────────────────────────────

SAMPLE_RATE = 24000  # Kokoro's native sample rate

def synthesize_chapter(
    pipeline,
    title: str,
    text: str,
    out_path: Path,
    voice: str = "af_heart",
    speed: float = 1.0,
) -> float:
    """
    Synthesize a single chapter to a WAV file.
    Returns duration in seconds.

    The chapter is split into natural sentence chunks before synthesis
    so that Kokoro can generate proper prosody per sentence group rather
    than trying to process thousands of words at once.
    """
    # Natural chapter introduction with a beat of silence after
    intro_text = f"Chapter. {title}."
    body_chunks = chunk_text(text, max_chars=400)

    print(f"    {len(body_chunks)} text chunks to synthesize...")

    audio_parts = []

    def synth(t: str):
        """Run Kokoro on a single text chunk, return audio array."""
        parts = []
        for _, _, audio in pipeline(t, voice=voice, speed=speed):
            parts.append(audio)
        return np.concatenate(parts) if parts else np.array([], dtype=np.float32)

    def silence(seconds: float):
        return np.zeros(int(SAMPLE_RATE * seconds), dtype=np.float32)

    # Chapter title announcement
    audio_parts.append(synth(intro_text))
    audio_parts.append(silence(1.2))  # Pause after title

    # Body
    for i, chunk in enumerate(body_chunks):
        if not chunk.strip():
            continue
        audio = synth(chunk)
        audio_parts.append(audio)

        # Natural inter-chunk pause (between sentence groups)
        audio_parts.append(silence(0.15))

        if (i + 1) % 25 == 0:
            pct = (i + 1) / len(body_chunks) * 100
            print(f"    {i+1}/{len(body_chunks)} chunks ({pct:.0f}%)")

    # Closing pause
    audio_parts.append(silence(1.5))

    full_audio = np.concatenate([a for a in audio_parts if len(a) > 0])
    sf.write(str(out_path), full_audio, SAMPLE_RATE, subtype='PCM_16')

    duration = len(full_audio) / SAMPLE_RATE
    print(f"    ✓ {out_path.name}  [{duration / 60:.1f} min]")
    return duration


# ── M4B export ────────────────────────────────────────────────────────────────

def export_m4b(
    chapter_files: list[Path],
    chapter_titles: list[str],
    chapter_durations: list[float],
    output_path: Path,
    book_title: str,
) -> None:
    """
    Combine per-chapter WAV files into a single M4B audiobook file
    with proper chapter markers. Requires ffmpeg in PATH.
    """
    if not shutil.which('ffmpeg'):
        print("\n⚠  ffmpeg not found — skipping M4B export.")
        print("   Install from https://ffmpeg.org and add to PATH to enable M4B output.")
        return

    print("\nBuilding M4B audiobook...")

    with tempfile.TemporaryDirectory() as tmp:
        tmp = Path(tmp)

        # 1. Write the ffmpeg concat list
        concat_file = tmp / "concat.txt"
        concat_file.write_text(
            "\n".join(f"file '{f.resolve()}'" for f in chapter_files),
            encoding='utf-8'
        )

        # 2. Concat all WAVs → single AAC
        combined_aac = tmp / "combined.aac"
        print("  Encoding audio (AAC)...")
        os.system(
            f'ffmpeg -y -f concat -safe 0 -i "{concat_file}" '
            f'-c:a aac -b:a 64k "{combined_aac}" -loglevel warning'
        )

        # 3. Build ffmpeg metadata file with chapter timestamps
        meta_file = tmp / "chapters.txt"
        lines = [";FFMETADATA1", f"title={book_title}", "artist=Kokoro TTS", ""]
        ts = 0
        for title, dur in zip(chapter_titles, chapter_durations):
            start_ms = int(ts * 1000)
            end_ms = int((ts + dur) * 1000)
            lines += [
                "[CHAPTER]",
                "TIMEBASE=1/1000",
                f"START={start_ms}",
                f"END={end_ms}",
                f"title={title}",
                "",
            ]
            ts += dur
        meta_file.write_text("\n".join(lines), encoding='utf-8')

        # 4. Wrap AAC in MP4 container with chapter metadata → .m4b
        print("  Writing M4B with chapter markers...")
        os.system(
            f'ffmpeg -y -i "{combined_aac}" -i "{meta_file}" '
            f'-map_metadata 1 -c copy "{output_path}" -loglevel warning'
        )

    if output_path.exists():
        size_mb = output_path.stat().st_size / 1_048_576
        print(f"  ✓ M4B saved: {output_path}  ({size_mb:.1f} MB)")
    else:
        print("  ✗ M4B export failed — check that ffmpeg is working correctly")


# ── Entry point ───────────────────────────────────────────────────────────────

VOICE_INFO = {
    # ── American English female ───────────────────────────────────────────────
    "af_heart":   "Warm, expressive, emotive — best all-round audiobook voice (American)",
    "af_alloy":   "Smooth, neutral, versatile — clean delivery, suits non-fiction (American)",
    "af_aoede":   "Soft, lyrical, musical cadence — great for poetry or literary fiction (American)",
    "af_bella":   "Bright, animated, youthful — engaging for YA or upbeat narratives (American)",
    "af_charlie": "Friendly, conversational, relaxed — natural everyday tone (American)",
    "af_jadzia":  "Energetic, vibrant, punchy — suits action or fast-paced stories (American)",
    "af_jessica": "Professional, articulate, composed — good for business or self-help (American)",
    "af_kore":    "Darker, deeper, mysterious — strong for thriller or gothic fiction (American)",
    "af_nicole":  "Calm, measured, soothing — well-paced, suits long-form narration (American)",
    "af_nova":    "Crisp, bright, confident — clear and attention-holding (American)",
    "af_river":   "Gentle, flowing, unhurried — natural storytelling warmth (American)",
    "af_sarah":   "Approachable, warm, clear — reliable all-purpose narrator (American)",
    "af_sky":     "Airy, light, breathy — soft delivery, suits contemplative or romance (American)",
    # ── American English male ─────────────────────────────────────────────────
    "am_adam":    "Warm, deep, trustworthy — classic audiobook narrator quality (American)",
    "am_echo":    "Resonant, smooth, deliberate — strong presence, suits drama (American)",
    "am_eric":    "Friendly, mid-range, approachable — natural and easy to listen to (American)",
    "am_fenrir":  "Deep, powerful, commanding — intense, suits action or fantasy (American)",
    "am_liam":    "Youthful, energetic, casual — good for contemporary or YA fiction (American)",
    "am_michael": "Clear, authoritative, measured — polished narrator, suits non-fiction (American)",
    "am_onyx":    "Rich, deep, velvety — smooth and immersive long-form voice (American)",
    "am_puck":    "Playful, quick, mischievous — expressive range, suits comedy or fantasy (American)",
    # ── British English female ────────────────────────────────────────────────
    "bf_alice":   "Refined, proper, precise — classic British female narrator (British)",
    "bf_emma":    "Elegant, composed, warm — versatile literary narrator (British)",
    "bf_lily":    "Soft, gentle, understated — calm and pleasant for long listens (British)",
    # ── British English male ──────────────────────────────────────────────────
    "bm_daniel":  "Professional, measured, even — reliable and clear documentary style (British)",
    "bm_fable":   "Rich, storytelling quality, slightly theatrical — strong for fiction (British)",
    "bm_george":  "Classic, authoritative narrator — traditional audiobook gravitas (British)",
    "bm_lewis":   "Deep, resonant, unhurried — commanding and immersive (British)",
}


def main():
    parser = argparse.ArgumentParser(
        description="Convert EPUB or PDF to an expressive AI audiobook using Kokoro TTS",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    parser.add_argument("input", nargs="?", help="Path to EPUB or PDF file")
    parser.add_argument(
        "-v", "--voice", default="af_heart",
        help="Kokoro voice ID (default: af_heart). Use --list-voices to see options."
    )
    parser.add_argument(
        "-s", "--speed", type=float, default=0.9,
        help="Narration speed multiplier. default 0.9 = slightly slower/more dramatic."
    )
    parser.add_argument(
        "-o", "--output", default=None,
        help="Output directory (default: <cwd>/<book_name>/)"
    )
    parser.add_argument(
        "-lv", "--list-voices", action="store_true",
        help="Print available voices and exit"
    )
    parser.add_argument(
        "--chapters-only", default=None,
        help="Comma-separated chapter indices to process, e.g. '1,2,3' (for testing)"
    )

    args = parser.parse_args()

    if args.list_voices:
        print("\nAvailable Kokoro voices:\n")
        for v, desc in VOICE_INFO.items():
            print(f"  {v:<14}  {desc}")
        print()
        return

    if not args.input:
        parser.print_help()
        sys.exit(1)

    input_path = Path(args.input)
    if not input_path.exists():
        print(f"Error: file not found: {input_path}")
        sys.exit(1)

    if input_path.suffix.lower() not in ('.epub', '.pdf'):
        print(f"Error: unsupported file type '{input_path.suffix}'. Use .epub or .pdf")
        sys.exit(1)

    out_dir = Path(args.output) if args.output else Path.cwd() / f"{input_path.stem}"
    out_dir.mkdir(parents=True, exist_ok=True)
    print(f"\n📚 Audiobook Maker")
    print(f"   Input : {input_path}")
    print(f"   Voice : {args.voice}  ({VOICE_INFO.get(args.voice, 'custom')})")
    print(f"   Speed : {args.speed}x")
    print(f"   Output: {out_dir}/\n")

    # ── Extract chapters ──────────────────────────────────────────────────────
    print("── Extracting chapters ──")
    if input_path.suffix.lower() == '.epub':
        chapters = extract_epub(input_path)
    else:
        chapters = extract_pdf(input_path)

    if not chapters:
        print("Error: no chapters extracted — check the file and try again.")
        sys.exit(1)

    # Optionally filter to specific chapters (useful for testing)
    if args.chapters_only:
        indices = [int(x) - 1 for x in args.chapters_only.split(',')]
        chapters = [chapters[i] for i in indices if i < len(chapters)]

    print(f"\nFound {len(chapters)} chapter(s):\n")
    for i, (title, text) in enumerate(chapters):
        word_count = len(text.split())
        est_min = word_count / (140 * args.speed)  # ~140 words/min narration
        print(f"  [{i+1:2d}] {title[:60]:<60}  {word_count:>6} words  (~{est_min:.0f} min)")

    total_words = sum(len(t.split()) for _, t in chapters)
    total_est = total_words / (140 * args.speed)
    print(f"\n  Total: {total_words:,} words  (~{total_est:.0f} min estimated)")

    # ── Load Kokoro ───────────────────────────────────────────────────────────
    print("\n── Loading Kokoro TTS (CUDA) ──")
    print("  (First run will download model weights ~330 MB)")
    try:
        import torch
        if not torch.cuda.is_available():
            print("\nError: CUDA is not available. Check that:")
            print("  1. You have an NVIDIA GPU")
            print("  2. PyTorch was installed with CUDA support:")
            print("     https://pytorch.org/get-started/locally/")
            sys.exit(1)
        gpu_name = torch.cuda.get_device_name(0)
        print(f"  ✓ GPU: {gpu_name}")

        from kokoro import KPipeline
        lang_code = 'b' if args.voice.startswith('b') else 'a'
        pipeline = KPipeline(lang_code=lang_code, device='cuda', repo_id='hexgrad/Kokoro-82M')
        print(f"  ✓ Kokoro loaded  (lang={lang_code}, device=cuda)\n")
    except ImportError:
        print("\nError: Kokoro not installed. Run:")
        print("  pip install kokoro soundfile")
        sys.exit(1)

    # ── Synthesize each chapter ───────────────────────────────────────────────
    print("── Synthesizing chapters ──\n")
    chapter_files = []
    chapter_durations = []

    for i, (title, text) in enumerate(chapters):
        # Sanitize title for filename
        safe = re.sub(r'[^\w\s\-]', '', title).strip()
        safe = re.sub(r'\s+', '_', safe)[:50]
        out_file = out_dir / f"{i+1:02d}_{safe}.wav"

        print(f"[{i+1}/{len(chapters)}] {title}")
        duration = synthesize_chapter(
            pipeline, title, text, out_file,
            voice=args.voice, speed=args.speed
        )
        chapter_files.append(out_file)
        chapter_durations.append(duration)

    # ── M4B (always) ──────────────────────────────────────────────────────────
    m4b_path = out_dir / f"{input_path.stem}_(kokoro_{args.voice}_{args.speed}).m4b"
    export_m4b(
        chapter_files,
        [t for t, _ in chapters],
        chapter_durations,
        m4b_path,
        book_title=input_path.stem.replace('_', ' ').replace('-', ' ').title(),
    )

    # ── Cleanup WAVs ──────────────────────────────────────────────────────────
    if m4b_path.exists():
        for f in chapter_files:
            f.unlink(missing_ok=True)

    # ── Summary ───────────────────────────────────────────────────────────────
    total_actual = sum(chapter_durations)
    print(f"\n{'═'*55}")
    print(f"✅  Done!")
    print(f"   Chapters : {len(chapter_files)}")
    print(f"   Runtime  : {total_actual/60:.1f} minutes")
    if m4b_path.exists():
        print(f"   M4B      : {m4b_path.resolve()}")
    print(f"{'═'*55}\n")


if __name__ == "__main__":
    main()
