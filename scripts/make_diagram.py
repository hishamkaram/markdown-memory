"""Draw the two theme variants of the README diagram from one description.

GitHub sanitises SVGs it renders from a repository and strips <style>, so a single file
that switched theme with a media query would render wrongly in one of the two. Two files,
presentation attributes only, selected with <picture> in the README.
"""

import os
import shutil
import subprocess
import sys
import tempfile
from pathlib import Path

FONT = (
    "ui-sans-serif,-apple-system,BlinkMacSystemFont,'Segoe UI',Roboto,"
    "'Helvetica Neue',Arial,sans-serif"
)
MONO = "ui-monospace,SFMono-Regular,Menlo,Consolas,'Liberation Mono',monospace"

THEMES = {
    "light": {
        "bg": "#ffffff",
        "ink": "#1f2328",
        "muted": "#59636e",
        "line": "#d1d9e0",
        "cardEdge": "#d1d9e0",
        "wash": "#f6f8fa",
        "cold": "#cf222e",
        "coldWash": "#fff1f0",
        "coldEdge": "#ffc1bc",
        "warm": "#1a7f37",
        "warmWash": "#eefbf1",
        "warmEdge": "#aee0bb",
        "accent": "#0969da",
        "accentWash": "#eff6ff",
        "accentEdge": "#b6d6fd",
    },
    "dark": {
        "bg": "#0d1117",
        "ink": "#e6edf3",
        "muted": "#9198a1",
        "line": "#30363d",
        "cardEdge": "#30363d",
        "wash": "#161b22",
        "cold": "#ff7b72",
        "coldWash": "#2d1618",
        "coldEdge": "#5c2b29",
        "warm": "#3fb950",
        "warmWash": "#12261a",
        "warmEdge": "#2d5c38",
        "accent": "#58a6ff",
        "accentWash": "#0f1d2e",
        "accentEdge": "#1f3d63",
    },
}

ROOT = Path(__file__).resolve().parent.parent

W, H = 1120, 430

# What reading each file whole costs, and what the query gives back instead. Both sides are
# measured, never estimated, and both are drawn to one scale - the largest file below - or
# the comparison this picture exists to make would be drawn dishonestly. No figure is
# repeated anywhere else in this file: every total, caption and label is derived from these
# two lists, because a number written out twice is a number that goes stale in one place.
# tests/test_agent_docs.py re-derives all of them from the real files, so editing the
# documentation without redrawing the picture is a test failure rather than a quiet lie.
LEFT_FILES = [
    ("README.md", 8876),
    ("CLAUDE.md", 4353),
    ("evaluation-protocol.md", 1661),
    ("AGENTS.md", 1298),
]
RIGHT_HITS = [
    (407, "What downloads, when, and where", True),
    (712, "Pre-download it, or install offline", False),
    (666, "markdown-memory  (preamble)", False),
    (738, "Commands  (CLAUDE.md)", False),
    (383, "What is checked before loading", False),
]
MAX_TOKENS = max(tokens for _, tokens in LEFT_FILES)
# Every figure the drawing prints is derived from the two lists above - the totals, the
# caption and the aria-label alike. Writing any of them out by hand is how the caption and
# the bar beside it come to disagree, which is precisely the defect this picture claims to
# be free of.
TOTAL_TOKENS = sum(tokens for _, tokens in LEFT_FILES)
# search_docs returns its first hit in full and the rest as pointers, so what comes back as
# text is the first hit; the others are drawn at what reading them would cost.
FULL_TOKENS = RIGHT_HITS[0][0]
POINTER_COUNT = len(RIGHT_HITS) - 1
# Counts read as words in prose, and prose is what the captions and the aria-label are.
_WORDS = ("no", "one", "two", "three", "four", "five", "six", "seven", "eight", "nine")


def spell(n: int) -> str:
    """`4` as "four". Past nine the digits read better anyway, and never raise."""
    return _WORDS[n] if 0 <= n < len(_WORDS) else str(n)


LEFT_BAR = 200.0
RIGHT_BAR = 200.0


def esc(text: str) -> str:
    return text.replace("&", "&amp;").replace("<", "&lt;").replace(">", "&gt;")


def text(x, y, s, *, fill, size=13, weight=400, anchor="start", font=FONT):
    return (
        f'<text x="{x}" y="{y}" font-family="{font}" font-size="{size}" '
        f'font-weight="{weight}" fill="{fill}" text-anchor="{anchor}">{esc(s)}</text>'
    )


def rect(x, y, w, h, *, fill, stroke, rx=8, sw=1):
    return (
        f'<rect x="{x}" y="{y}" width="{w}" height="{h}" rx="{rx}" '
        f'fill="{fill}" stroke="{stroke}" stroke-width="{sw}"/>'
    )


def arrow(x0, x1, y, colour):
    return (
        f'<path d="M{x0} {y} L{x1} {y}" stroke="{colour}" stroke-width="2" fill="none"/>'
        f'<path d="M{x1 - 7} {y - 5} L{x1} {y} L{x1 - 7} {y + 5}" fill="none" '
        f'stroke="{colour}" stroke-width="2"/>'
    )


def draw(c: dict) -> str:
    o = [
        f'<svg xmlns="http://www.w3.org/2000/svg" viewBox="0 0 {W} {H}" width="{W}" '
        f'height="{H}" role="img" aria-label="One question: reading {spell(len(LEFT_FILES))} '
        f"whole files costs {TOTAL_TOKENS:,} tokens; markdown-memory returns the section "
        f"that answers in full, {FULL_TOKENS} tokens, with {spell(POINTER_COUNT)} pointers "
        f'to the rest.">'
    ]
    o.append(f'<rect width="{W}" height="{H}" fill="{c["bg"]}"/>')

    o.append(
        text(
            40,
            42,
            '"where does the embedding model get downloaded?"',
            fill=c["ink"],
            size=17,
            weight=600,
            font=MONO,
        )
    )
    o.append(
        text(
            40,
            64,
            f"one question, asked of {spell(len(LEFT_FILES))} documentation files",
            fill=c["muted"],
            size=12,
        )
    )

    # ---- left: reading the files whole ----
    o.append(rect(40, 88, 310, 274, fill=c["coldWash"], stroke=c["coldEdge"]))
    o.append(text(62, 116, "Read the files", fill=c["cold"], size=13, weight=600))
    files = LEFT_FILES
    y = 138
    for name, tokens in files:
        o.append(text(62, y, name, fill=c["ink"], size=11, font=MONO))
        o.append(
            rect(
                62,
                y + 8,
                round(tokens * LEFT_BAR / MAX_TOKENS),
                13,
                fill=c["cold"],
                stroke="none",
                rx=3,
            )
        )
        o.append(
            text(328, y + 19, f"{tokens:,}", fill=c["muted"], size=11, anchor="end", font=MONO)
        )
        y += 46
    o.append(f'<path d="M62 318 L328 318" stroke="{c["coldEdge"]}" stroke-width="1"/>')
    o.append(text(62, 338, f"{TOTAL_TOKENS:,} tokens", fill=c["cold"], size=14, weight=600))
    o.append(text(62, 354, "most of it about something else", fill=c["muted"], size=11))

    # ---- middle: what it does with them ----
    o.append(text(430, 116, "markdown-memory", fill=c["ink"], size=13, weight=600))
    o.append(rect(414, 132, 232, 38, fill=c["wash"], stroke=c["cardEdge"]))
    o.append(text(530, 156, "split at every heading", fill=c["ink"], size=12, anchor="middle"))
    o.append(rect(414, 190, 110, 44, fill=c["accentWash"], stroke=c["accentEdge"]))
    o.append(text(469, 210, "keywords", fill=c["accent"], size=11, anchor="middle"))
    o.append(text(469, 225, "FTS5 / BM25", fill=c["muted"], size=10, anchor="middle", font=MONO))
    o.append(rect(536, 190, 110, 44, fill=c["accentWash"], stroke=c["accentEdge"]))
    o.append(text(591, 210, "vectors", fill=c["accent"], size=11, anchor="middle"))
    o.append(
        text(591, 225, "section + passage", fill=c["muted"], size=9, anchor="middle", font=MONO)
    )
    o.append(rect(414, 254, 232, 38, fill=c["wash"], stroke=c["cardEdge"]))
    o.append(text(530, 278, "fused, best rank wins (RRF)", fill=c["ink"], size=12, anchor="middle"))
    # The split forks into two rankings and they rejoin at the fusion: drawn as a fork,
    # because a bare vertical stub reads as a line that goes nowhere.
    o.append(
        f'<path d="M530 170 L530 180 M469 180 L591 180 M469 180 L469 190 '
        f'M591 180 L591 190" stroke="{c["cardEdge"]}" stroke-width="2" fill="none"/>'
    )
    o.append(
        f'<path d="M469 234 L469 244 M591 234 L591 244 M469 244 L591 244 '
        f'M530 244 L530 254" stroke="{c["accentEdge"]}" stroke-width="2" fill="none"/>'
    )
    o.append(
        text(
            530,
            318,
            "on your CPU; nothing leaves the machine",
            fill=c["muted"],
            size=11,
            anchor="middle",
        )
    )

    o.append(arrow(358, 406, 212, c["line"]))
    o.append(arrow(654, 702, 212, c["line"]))

    # ---- right: what comes back ----
    o.append(rect(710, 88, 370, 274, fill=c["warmWash"], stroke=c["warmEdge"]))
    o.append(
        text(
            732,
            116,
            f"One section in full, {spell(POINTER_COUNT)} pointers",
            fill=c["warm"],
            size=13,
            weight=600,
        )
    )
    hits = RIGHT_HITS
    y = 140
    for rank, (tokens, label, best) in enumerate(hits):
        # Solid: text that comes back. Outlined: a pointer, and what following it would cost.
        o.append(
            rect(
                732,
                y,
                max(3, round(tokens * RIGHT_BAR / MAX_TOKENS)),
                11,
                fill=c["warm"] if rank == 0 else "none",
                stroke="none" if rank == 0 else c["warm"],
                rx=2,
            )
        )
        o.append(text(776, y + 10, f"{tokens}", fill=c["muted"], size=11, anchor="end", font=MONO))
        o.append(
            text(
                788,
                y + 10,
                label,
                fill=c["ink"] if best else c["muted"],
                size=11,
                weight=600 if best else 400,
                font=MONO,
            )
        )
        y += 34
    o.append(f'<path d="M732 318 L1058 318" stroke="{c["warmEdge"]}" stroke-width="1"/>')
    o.append(text(732, 338, f"{FULL_TOKENS} tokens in full", fill=c["warm"], size=14, weight=600))
    o.append(
        text(
            732,
            354,
            f"plus {spell(POINTER_COUNT)} pointers, read only if the first is not the answer",
            fill=c["muted"],
            size=11,
        )
    )

    o.append(
        text(
            40,
            402,
            "Measured on this repository's own documentation. Token "
            "estimates; every bar is to the same scale.",
            fill=c["muted"],
            size=11,
        )
    )
    o.append("</svg>")
    return "\n".join(o) + "\n"


# The README offers the SVGs as <source>s and points its <img> - the element every client
# understands - at a PNG, so a reader whose client does not implement <picture> still sees
# the drawing. Rasterised at 2x so it stays sharp on a phone.
#
# This is not what fixes the GitHub mobile app, which draws a broken-image mark for this
# README today. The app renders a relative image path fine in a public repository; what it
# will not do is the authenticated fetch a private one needs. Two changes of format here
# moved nothing, and a third would not either.
PNG_SCALE = 2
BROWSERS = ("chromium", "chromium-browser", "google-chrome", "google-chrome-stable")


def rasterise(svg: Path, png: Path) -> bool:
    """Render `svg` to `png` with headless Chrome. False when no browser is installed."""
    browser = next((shutil.which(name) for name in BROWSERS if shutil.which(name)), None)
    if browser is None:
        return False
    # --no-sandbox only where the sandbox cannot work anyway: as root, Chrome refuses to
    # start without it. Passing it unconditionally would drop the sandbox on the machine
    # of everyone who runs this as themselves, to fix a problem they do not have.
    sandbox = ["--no-sandbox"] if os.geteuid() == 0 else []
    with tempfile.TemporaryDirectory() as scratch:
        done = subprocess.run(
            [browser, "--headless", "--disable-gpu", *sandbox, "--hide-scrollbars",
             f"--force-device-scale-factor={PNG_SCALE}", f"--screenshot={png}",
             f"--window-size={W},{H}", f"--user-data-dir={scratch}", svg.as_uri()],
            capture_output=True, text=True,
        )  # fmt: skip
    if done.returncode != 0:
        # Report and carry on rather than exiting here: main() draws both themes, and dying
        # inside the first one leaves the second theme's SVG and PNG disagreeing too - one
        # stale pair turned into two. Chrome says why on stderr; swallowing it would leave
        # "it did not work" and nothing else.
        print(
            f"{browser} failed to rasterise {svg.name}:\n{done.stderr.strip()}",
            file=sys.stderr,
        )
        return False
    return True


def main() -> int:
    missing = False
    for name, colours in THEMES.items():
        path = ROOT / "docs/assets" / f"how-it-works-{name}.svg"
        path.write_text(draw(colours), encoding="utf-8")
        print(path.relative_to(ROOT))
        png = path.with_suffix(".png")
        if rasterise(path, png):
            print(png.relative_to(ROOT))
        else:
            print(f"  {png.name} was NOT redrawn", file=sys.stderr)
            missing = True
    if missing:
        # The SVGs above have already been rewritten, so exiting 0 here would hand back a
        # vector carrying new figures and a raster still carrying the old ones - the exact
        # disagreement between the two images that the fallback exists to avoid. Say so
        # loudly: the drawing is only half regenerated until a browser is installed.
        print(
            "install a headless chromium and re-run: the PNGs the README falls back to are "
            "now stale",
            file=sys.stderr,
        )
        return 1
    return 0


# Guarded, and not merely for tidiness: tests/test_agent_docs.py imports this module to
# read the figures it draws. Writing at import time would have that test regenerate the
# SVGs it then checks, which is an assertion that cannot fail - and a test run that edits
# tracked files.
if __name__ == "__main__":
    raise SystemExit(main())
