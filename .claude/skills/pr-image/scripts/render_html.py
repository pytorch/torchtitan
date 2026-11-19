#!/usr/bin/env python3
# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""Render a pr-image HTML page to a PNG: 900 CSS px wide, 2x pixels, cropped.

    python3 -I render_html.py <folder>/lanes.html
    # writes lanes.png (1800 px wide) and lanes_900px.png (the size GitHub shows)

    python3 -I render_html.py pr_image/lanes.html --upload
    # GitHub Actions: renders again, checks it equals lanes.png, uploads, prints the URL

The page should set `body { margin: 0; width: 900px; background: <opaque colour> }`:
the crop keeps everything above the last row that differs from the bottom-left pixel.

The page can't read local files or reach the network, so a drawing can't show what
isn't in its own HTML:
1. Chrome loads it as a `data:` URL, so `file://` iframes and images stay blank.
2. A Content-Security-Policy tag allows only inline styles, scripts, and `data:` images and fonts.
3. Chrome resolves no host names and sends no WebRTC UDP packets, which also stops page
   navigations and STUN requests (CSP covers neither). Its env has only PATH and an
   empty HOME.
"""

# no-test: needs headless Chrome, and CI runs only tests/. Check it by rendering the 3
# files in ../examples/ and looking at each PNG. Then check that it refuses a symlink
# x.html -> /proc/self/environ, and an --upload after the page changed.

import argparse
import base64
import glob
import json
import os
import re
import shutil
import subprocess
import tempfile
from pathlib import Path

from PIL import Image, ImageChops

WIDTH_CSS_PX = 900
SCALE = 2
PLAYWRIGHT_CHROME_GLOB = (
    "~/.cache/ms-playwright/chromium_headless_shell-*/*/chrome-headless-shell"
)
CHROME_NAMES = ("google-chrome", "chromium", "chromium-browser")

CSP_TAG = (
    '<meta http-equiv="Content-Security-Policy" content="default-src \'none\'; '
    "style-src 'unsafe-inline'; script-src 'unsafe-inline'; img-src data:; font-src data:\">"
)
# The CSP tag goes right after the doctype: any tag before the doctype puts Chrome in
# quirks mode, which shifts layouts. Each leading comment ends at its first `-->`, so
# a page of comments with no doctype can't make the match backtrack for minutes.
LEADING_DOCTYPE = re.compile(
    r"\A(\s*(?:<!--(?:(?!-->).)*-->\s*)*<!doctype html[^>]*>)",
    re.IGNORECASE | re.DOTALL,
)
# Linux caps one command-line argument at 128 KiB, and Chrome gets the page as one.
MAX_PAGE_URL_BYTES = 120 * 1024

BUCKET = "ossci-raw-job-status"
# The page's file name becomes part of the S3 key.
IMAGE_STEM = re.compile(r"[a-z0-9_-]{1,40}")
PAD_CSS_PX = 24
# The examples render in under a second; a page that keeps reloading stops here.
CHROME_TIMEOUT_S = 30


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("html", type=Path)
    parser.add_argument(
        "--upload",
        action="store_true",
        help="GitHub Actions: render again, upload, print the URL",
    )
    parser.add_argument(
        "--max-height",
        type=int,
        default=3000,
        help="page height in CSS px before the crop",
    )
    args = parser.parse_args()

    # The name becomes part of the S3 key: check it before any render.
    if not IMAGE_STEM.fullmatch(args.html.stem):
        raise ValueError(
            f"name the page {IMAGE_STEM.pattern}.html; got {args.html.name}"
        )

    # A PR can plant a symlink such as x.html -> /proc/self/environ: refuse a path
    # that doesn't resolve to an .html file.
    html = args.html.resolve()
    if html.suffix != ".html" or not html.is_file():
        raise ValueError(f"{args.html} must be an .html file; it resolves to {html}")
    if args.upload and not html.is_relative_to(Path.cwd().resolve()):
        raise ValueError(
            f"--upload renders pages inside {Path.cwd()} only; {args.html} resolves to {html}"
        )

    page = html.read_text(encoding="utf-8")
    doctype = LEADING_DOCTYPE.match(page)
    if doctype:
        page = doctype.group(1) + CSP_TAG + page[doctype.end() :]
    else:
        page = CSP_TAG + page
    page_url = (
        "data:text/html;charset=utf-8;base64,"
        + base64.b64encode(page.encode()).decode()
    )
    if len(page_url) > MAX_PAGE_URL_BYTES:
        raise ValueError(
            f"{args.html} is {len(page_url)} bytes as a data: URL; the limit is {MAX_PAGE_URL_BYTES}"
        )

    chrome = _find_chrome()
    png = args.html.absolute().with_suffix(".png")
    if not args.upload:
        image = _render(chrome, page_url, png, max_height=args.max_height)
        preview = png.with_name(f"{png.stem}_900px.png")
        preview.unlink(missing_ok=True)
        image.resize(
            (WIDTH_CSS_PX, image.height // SCALE), Image.Resampling.LANCZOS
        ).save(preview)
        print(f"{png} {image.width}x{image.height}, check {preview}")
        return

    # Upload a fresh render that matches the PNG Claude looked at, never a file found
    # in the checkout.
    checked = Image.open(png).convert("RGB")
    with tempfile.TemporaryDirectory() as temp_dir:
        fresh_png = Path(temp_dir) / png.name
        fresh = _render(chrome, page_url, fresh_png, max_height=args.max_height)
        if (
            fresh.size != checked.size
            or ImageChops.difference(fresh, checked).getbbox() is not None
        ):
            raise RuntimeError(
                f"{args.html} no longer renders as {png}: render it again and check it before --upload"
            )
        print(_upload(fresh_png, stem=png.stem))


def _find_chrome() -> str:
    """Return a cached Playwright headless Chrome, else the first Chrome on PATH."""
    cached = sorted(glob.glob(os.path.expanduser(PLAYWRIGHT_CHROME_GLOB)))
    if cached:
        return cached[-1]
    for name in CHROME_NAMES:
        found = shutil.which(name)
        if found:
            return found
    raise RuntimeError(
        f"no headless Chrome: nothing at {PLAYWRIGHT_CHROME_GLOB} and none of {CHROME_NAMES} on PATH. "
        "Run `npx playwright install chromium-headless-shell`."
    )


def _render(chrome: str, page_url: str, png: Path, max_height: int) -> Image.Image:
    """Screenshot `page_url` into `png`, crop it below the content, and return it."""
    # Delete first: a symlink a PR planted at this path would make the screenshot
    # overwrite its target.
    png.unlink(missing_ok=True)
    with tempfile.TemporaryDirectory() as empty_home:
        try:
            chrome_run = subprocess.run(
                [
                    chrome,
                    "--headless",
                    "--no-sandbox",
                    "--disable-gpu",
                    "--hide-scrollbars",
                    "--host-resolver-rules=MAP * ~NOTFOUND",
                    "--force-webrtc-ip-handling-policy=disable_non_proxied_udp",
                    f"--force-device-scale-factor={SCALE}",
                    f"--window-size={WIDTH_CSS_PX},{max_height}",
                    f"--screenshot={png}",
                    page_url,
                ],
                capture_output=True,
                text=True,
                timeout=CHROME_TIMEOUT_S,
                env={"PATH": os.environ["PATH"], "HOME": empty_home},
            )
        except subprocess.TimeoutExpired:
            raise RuntimeError(
                f"Chrome took over {CHROME_TIMEOUT_S} s to render {png.stem}; does the page reload?"
            ) from None
    # Raise with Chrome's stderr only: its command line holds the whole page URL.
    if chrome_run.returncode != 0:
        stderr_tail = "\n".join(chrome_run.stderr.strip().splitlines()[-5:])
        raise RuntimeError(f"Chrome exited with {chrome_run.returncode}: {stderr_tail}")

    image = Image.open(png).convert("RGB")
    background = Image.new("RGB", image.size, image.getpixel((0, image.height - 1)))
    content_box = ImageChops.difference(image, background).getbbox()
    if content_box is None:
        raise ValueError(f"{png} is blank")
    if content_box[3] >= image.height - SCALE:
        raise ValueError(
            f"content reaches the bottom of a {max_height} px page; pass a larger --max-height"
        )
    cropped = image.crop(
        (0, 0, image.width, min(image.height, content_box[3] + PAD_CSS_PX * SCALE))
    )
    cropped.save(png)
    return cropped


def _upload(png: Path, stem: str) -> str:
    """Upload `png` under the workflow's public review-logs prefix; return its URL.

    Example, run 44437007501 attempt 1 on PR 6181, page pr_image/lanes.html:
        review-logs/claude-code/pytorch/torchtitan/6181/images/44437007501_1_lanes.png
    """
    issue = json.loads(Path(os.environ["GITHUB_EVENT_PATH"]).read_text()).get("issue")
    if issue is None:
        raise ValueError(
            f"{os.environ['GITHUB_EVENT_PATH']} has no issue: --upload runs on @claude comments only"
        )

    key = (
        f"review-logs/claude-code/{os.environ['GITHUB_REPOSITORY']}/{issue['number']}/images/"
        f"{os.environ['GITHUB_RUN_ID']}_{os.environ['GITHUB_RUN_ATTEMPT']}_{stem}.png"
    )
    # An empty HOME, so no ~/.aws/cli/alias can turn `s3 cp` into another command. The
    # credentials and region come from the AWS_* env.
    with tempfile.TemporaryDirectory() as empty_home:
        copy = subprocess.run(
            [
                "aws",
                "s3",
                "cp",
                str(png),
                f"s3://{BUCKET}/{key}",
                "--content-type",
                "image/png",
                "--no-progress",
            ],
            capture_output=True,
            text=True,
            timeout=120,
            env={**os.environ, "HOME": empty_home},
        )
    if copy.returncode != 0:
        raise RuntimeError(f"aws s3 cp of {png} to {key} failed: {copy.stderr.strip()}")
    return f"https://{BUCKET}.s3.amazonaws.com/{key}"


if __name__ == "__main__":
    main()
