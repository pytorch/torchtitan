# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

import json
import os
import re
import subprocess
import sys

# Regex to extract GitHub PR/Issue links: captures URL, Owner, Repo, Type (pull/issues), and Number
LINK_REGEX = re.compile(
    r"(https://github\.com/([\w\-\.]+)/([\w\-\.]+)/(pull|issues)/(\d+))"
)


def get_repo_full_name() -> str:
    """
    Retrieves the 'owner/repo' string.

    Priority: Environment variable -> Local Git config.

    Returns:
        str: The full repository name (e.g., 'pytorch/torchtitan').
    """
    repo = os.environ.get("GITHUB_REPOSITORY")
    if not repo:
        try:
            # Fallback for local testing: parse origin URL
            url = subprocess.run(
                ["git", "remote", "get-url", "origin"], capture_output=True, text=True
            ).stdout.strip()
            match = re.search(r"github\.com[:/](.+?)(?:\.git)?$", url)
            repo = match.group(1) if match else "owner/repo"
        except Exception:
            repo = "owner/repo"
    return repo


def get_git_info() -> tuple[str, str]:
    """
    Retrieves repository metadata for constructing permalinks.

    Returns:
        tuple[str, str]: A tuple containing (repo_full_name, branch_or_sha).
    """
    repo = get_repo_full_name()
    # Use branch name (ref_name) if available, otherwise fallback to main
    ref = os.environ.get("GITHUB_REF_NAME", "main")
    return repo, ref


def get_line_author(file_path: str, line_no: int) -> str:
    """
    Uses 'git blame' to identify the author of a specific line.

    Args:
        file_path: Path to the file.
        line_no: The specific line number to blame.

    Returns:
        str: The author's name or 'Unknown'.
    """
    try:
        # --porcelain provides machine-readable output where 'author ' is a stable prefix
        cmd = ["git", "blame", "-L", f"{line_no},{line_no}", "--porcelain", file_path]
        res = subprocess.run(cmd, capture_output=True, text=True)
        prefix = "author "
        for line in res.stdout.splitlines():
            if line.startswith(prefix):
                return line[len(prefix) :].strip()
    except Exception:
        return "Unknown"
    return "Unknown"


def get_github_data(
    owner: str, repo: str, api_type: str, number: str
) -> tuple[str, str]:
    """
    Queries the GitHub API via the 'gh' CLI for item status and title.

    Args:
        owner: Repo owner.
        repo: Repo name.
        api_type: 'pulls' or 'issues'.
        number: The ID of the PR or Issue.

    Returns:
        tuple[str, str]: (status, title). Status is normalized to 'merged', 'closed', or 'open'.
    """
    endpoint = f"repos/{owner}/{repo}/{api_type}/{number}"
    cmd = [
        "gh",
        "api",
        endpoint,
        "--jq",
        "{state: .state, merged: .merged, title: .title}",
    ]
    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(res.stdout)

        status = data.get("state", "unknown")
        # PRs can be 'closed' but 'merged'; we prioritize 'merged' as a distinct state
        if api_type == "pulls" and data.get("merged"):
            status = "merged"

        return status, data.get("title", "No title found")
    except Exception:
        return "unknown", "Unknown Title"


def parse_comment_block(
    block_lines: list[str], start_line_no: int, file_path: str
) -> dict[str, str] | None:
    """
    Validates a block of comments and extracts TODOs details if a GitHub link is present.

    Args:
        block_lines: List of raw comment lines.
        start_line_no: Line number where the block started.
        file_path: Relative path to the file.

    Returns:
        dict | None: Dictionary of todo metadata or None if invalid.
    """
    full_text = " ".join(block_lines)
    match = LINK_REGEX.search(full_text)

    if "TODO" in full_text and match:
        url, owner, repo_name, item_type, number = match.groups()
        api_type = "pulls" if item_type == "pull" else "issues"

        status, title = get_github_data(owner, repo_name, api_type, number)
        repo_path, ref = get_git_info()
        author = get_line_author(file_path, start_line_no)

        # Permalink points to the specific line on GitHub
        location_url = (
            f"https://github.com/{repo_path}/blob/{ref}/{file_path}#L{start_line_no}"
        )

        # Strip '#' and leading/trailing whitespace from each line in the block
        clean_text = " ".join([l.lstrip().lstrip("#").strip() for l in block_lines])

        return {
            "status": status,
            "title": title,
            "text": clean_text,
            "author": author,
            "file_location_label": f"{file_path}:{start_line_no}",
            "file_location_url": location_url,
        }
    return None


def scan_file_for_todos(file_path: str) -> list[dict[str, str]]:
    """
    Scans a Python file for contiguous comment blocks containing TODOs.

    Args:
        file_path: Path to the .py file.

    Returns:
        List[dict]: A list of detected todo objects.
    """
    found = []
    try:
        with open(file_path, "r", encoding="utf-8") as f:
            lines = f.readlines()
            current_block, block_start = [], 0

            for i, line in enumerate(lines, start=1):
                stripped = line.lstrip()
                is_comment = stripped.startswith("#")
                is_new_todo = "TODO" in stripped

                # Logic: End current block if we hit code OR a brand new todo line
                if not is_comment or (is_new_todo and current_block):
                    if current_block:
                        item = parse_comment_block(
                            current_block, block_start, file_path
                        )
                        if item:
                            found.append(item)
                    current_block = []

                if is_comment:
                    if not current_block:
                        block_start = i
                    current_block.append(line.rstrip())

            # Catch trailing blocks at the end of the file
            if current_block:
                item = parse_comment_block(current_block, block_start, file_path)
                if item:
                    found.append(item)
    except Exception as e:
        print(f"Error reading {file_path}: {e}", file=sys.stderr)
    return found


def generate_markdown_report(all_items: list[dict[str, str]]) -> str:
    """
    Constructs a formatted Markdown report with emojis and layout fixes.

    Args:
        all_items: List of all detected TODOs across the repo.

    Returns:
        str: The final Markdown string.
    """
    groups = {"merged": [], "closed": [], "open": [], "unknown": []}
    for item in all_items:
        groups.get(item["status"], groups["unknown"]).append(item)

    emojis = {"merged": "💜 Merged", "closed": "🔴 Closed", "open": "🟢 Open"}
    output = "## 🚀 GitHub TODO Status Report\n\n"
    # A one-paragraph summary
    output += (
        "This report automatically scans the codebase for comment blocks that start with `TODO` "
        "and contain a link to a GitHub Pull Request or Issue.</br> It tracks the current status of "
        "those references (Open, Closed, or Merged) to help identify technical debt that is ready "
        "to be addressed.\n\n"
    )
    has_any = False

    for status in ("merged", "closed", "open"):
        if groups[status]:
            has_any = True
            output += f"### {emojis[status]}\n"
            output += "| Author | TODO Description | Issue/PR Title | File Location |\n"
            output += "| :--- | :--- | :--- | :--- |\n"
            for item in groups[status]:
                label = item["file_location_label"]
                # 1. Reverse the string
                # 2. Replace the first '/' (originally the last one) with '>rb<' plus the slash
                # 3. Reverse it back to normal
                if "/" in label:
                    file_location_label = label[::-1].replace("/", ">rb</", 1)[::-1]
                else:
                    file_location_label = label
                # Escape underscores to prevent Markdown from bolding '__init__'
                file_location_label = file_location_label.replace("__", r"\_\_")

                output += (
                    f"| **{item['author']}** | {item['text']} | {item['title']} | "
                    f"[{file_location_label}]({item['file_location_url']}) |\n"
                )

            output += "\n"

    if not has_any:
        output += "✅ No tracked TODOs found."

    return output


def main() -> None:
    """
    Main orchestration function: finds files, scans them, and writes the report.
    """
    # Use git ls-files to respect .gitignore and skip hidden directories
    res = subprocess.run(
        ["git", "ls-files", "--cached", "--others", "--exclude-standard"],
        capture_output=True,
        text=True,
    )
    files = [f for f in res.stdout.splitlines() if f.endswith(".py")]

    all_found = []
    for f in files:
        all_found.extend(scan_file_for_todos(f))
    report = generate_markdown_report(all_found)

    # GITHUB_STEP_SUMMARY is a special file provided by GitHub Actions for job reports
    summary_path = os.environ.get("GITHUB_STEP_SUMMARY")
    if summary_path:
        with open(summary_path, "a") as f:
            f.write(report)
    else:
        print(report)


if __name__ == "__main__":
    main()
