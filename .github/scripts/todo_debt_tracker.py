# Copyright (c) Meta Platforms, Inc. and affiliates.
# All rights reserved.
#
# This source code is licensed under the BSD-style license found in the
# LICENSE file in the root directory of this source tree.

"""
This code automatically scans the codebase for comment blocks that start with todo
and contain a link to a GitHub Pull Request or an Issue.
It tracks the current status of those references (Open, Closed, or Merged)
to help identify technical debt that is ready to be addressed.
"""

import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone

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
        # 1. Get the commit hash for the specific line
        # The first line of porcelain output is the hash
        cmd_blame = [
            "git",
            "blame",
            "-L",
            f"{line_no},{line_no}",
            "--porcelain",
            file_path,
        ]
        res_blame = subprocess.run(
            cmd_blame, capture_output=True, text=True, check=True
        )

        first_line = res_blame.stdout.splitlines()[0]
        commit_hash = first_line.split()[0]

        # Handle 'not committed yet' local changes
        if not commit_hash or all(c == "0" for c in commit_hash):
            return "@you (uncommitted)"

        # 2. Ask GitHub API for the login associated with this commit
        # Using a simple JQ filter to get the author's login
        repo_full_name = get_repo_full_name()
        cmd_gh = [
            "gh",
            "api",
            f"repos/{repo_full_name}/commits/{commit_hash}",
            "--jq",
            ".author.login // .committer.login",
        ]

        res_gh = subprocess.run(cmd_gh, capture_output=True, text=True, check=True)
        login = res_gh.stdout.strip()

        # Handle empty/null response from JQ in Python instead of inside the JQ string
        if not login or login == "null":
            return "Unknown"

        return f"@{login}"

    except Exception as e:
        return "Unknown"


def get_github_data(
    owner: str, repo: str, api_type: str, number: str
) -> dict[str, str]:
    """
    Fetches lifecycle metadata for a GitHub Issue or Pull Request using the 'gh' CLI.

    Args:
        owner: The owner/organization of the repository.
        repo: The repository name.
        api_type: The GitHub API endpoint type ('pulls' or 'issues').
        number: The specific Issue or Pull Request number.

    Returns:
        A dictionary containing:
            - 'status': Normalized state ('open', 'closed', or 'merged').
            - 'event_timestamp': The ISO date of the most recent significant event (merged_at or closed_at).
            - 'created_timestamp': The ISO date the item was opened.
        Returns 'unknown' status and empty timestamps on failure.
    """

    endpoint = f"repos/{owner}/{repo}/{api_type}/{number}"
    cmd = [
        "gh",
        "api",
        endpoint,
        "--jq",
        "{state: .state, merged: .merged, merged_at: .merged_at, closed_at: .closed_at, created_at: .created_at}",
    ]

    try:
        res = subprocess.run(cmd, capture_output=True, text=True)
        data = json.loads(res.stdout)

        status = data.get("state", "unknown")
        # PRs can be 'closed' but 'merged'; prioritize 'merged' as a distinct state
        if api_type == "pulls" and data.get("merged"):
            status = "merged"

        # Determine which timestamp to use
        event_iso = (
            data.get("merged_at") if data.get("merged") else data.get("closed_at")
        )
        created_iso = data.get("created_at")

        return {
            "status": status,
            "event_timestamp": event_iso,
            "created_timestamp": created_iso,
        }
    except Exception:
        return {
            "status": "unknown",
            "event_timestamp": "",
            "created_timestamp": "",
        }


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
        url, owner, repo_name, item_type, number_str = match.groups()
        api_type = "pulls" if item_type == "pull" else "issues"

        result_dict = get_github_data(owner, repo_name, api_type, number_str)
        repo_path, ref = get_git_info()
        author = get_line_author(file_path, start_line_no)

        # Permalink points to the specific line on GitHub
        location_url = (
            f"https://github.com/{repo_path}/blob/{ref}/{file_path}#L{start_line_no}"
        )

        # Strip '#' and leading/trailing whitespace from each line in the block
        clean_text = " ".join([l.lstrip().lstrip("#").strip() for l in block_lines])

        return {
            "comment_author": author,
            "comment_text": clean_text,
            "blocker_status": result_dict["status"],
            "blocker_created_at": result_dict["created_timestamp"],
            "blocker_event_at": result_dict["event_timestamp"],
            "file_location_label": f"{file_path}:{start_line_no}",
            "file_location_url": location_url,
            "blocker_github_number": number_str,
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


def get_human_readable_time(iso_date_str: str) -> str:
    """
    Converts an ISO date string into a human-friendly relative time string.

    Args:
        iso_date_str: The ISO timestamp string (e.g., from GitHub API).
                      Handles 'Z' suffix by converting to +00:00 offset.

    Returns:
        A string representing the relative time: 'today', 'yesterday',
        'X days ago', or 'unknown' if the input is empty.
    """

    if not iso_date_str:
        return "unknown"

    # GitHub returns UTC times
    past_date = datetime.fromisoformat(iso_date_str.replace("Z", "+00:00"))
    now = datetime.now(timezone.utc)
    diff = now - past_date

    if diff.days == 0:
        return "today"
    elif diff.days == 1:
        return "yesterday"
    else:
        # Use non-breaking spaces to prevent vertical stacking in tables
        return f"{diff.days} days ago"


def generate_markdown_report(all_items: list[dict[str, str]]) -> str:
    """
    Constructs a formatted Markdown report with emojis and layout fixes.

    Args:
        all_items: List of all detected TODOs across the repo.

    Returns:
        str: The final Markdown string.
    """

    output = "## 🤖 Automatic TODO Status Report\n\n"
    # A one-paragraph summary
    output += (
        "This report automatically scans the codebase for comment blocks that start with `TODO` "
        "and contain a link to a GitHub Pull Request or an Issue.</br> It tracks the current status of "
        "those references (Open, Closed, or Merged) to help identify technical debt that is ready "
        "to be addressed.\n\n"
    )

    groups = {"merged": [], "closed": [], "open": [], "unknown": []}
    for item in all_items:
        groups.get(item["blocker_status"], groups["unknown"]).append(item)

    emojis = {"merged": "💜 Merged", "closed": "🔴 Closed", "open": "🟢 Open"}
    has_any = False

    for status in ("merged", "closed", "open"):
        if groups[status]:
            has_any = True
            output += f"### {emojis[status]}\n"
            status_title = "opened" if status == "open" else status
            output += f"| Comment Author | Comment Content | Blocker {status_title.capitalize()} | Comment Location |\n"
            output += "| :--- | :--- | :--- | :--- |\n"

            sorting_timestamp_key = (
                "blocker_created_at" if status == "open" else "blocker_event_at"
            )
            sorted_items = sorted(
                groups[status],
                key=lambda x: (
                    x[sorting_timestamp_key] or "2001-01-01",
                    int(x["blocker_github_number"]),
                ),
            )

            for item in sorted_items:

                # 1. Comment author
                author = item["comment_author"]

                # 2. Comment text of the todo
                description = item["comment_text"]

                # 3. Timestamp of the PR/Issue
                timestamp = (
                    item["blocker_created_at"]
                    if status == "open"
                    else item["blocker_event_at"]
                )
                timestamp = get_human_readable_time(timestamp)

                # 4. File where todo comment is located
                # 4.1. Label
                label = item["file_location_label"]
                # Allow browser to line split filename and line number from the rest
                if "/" in label:
                    file_location_label = label[::-1].replace("/", ">rb</", 1)[::-1]
                else:
                    file_location_label = label
                # Escape underscores to prevent Markdown from bolding '__init__'
                file_location_label = file_location_label.replace("__", r"\_\_")
                # 4.2. Link
                file_location_link = item["file_location_url"]

                # 5. Write into the table
                output += f"| **{author}** | {description} | {timestamp} | [{file_location_label}]({file_location_link}) |\n"

            output += "\n"

    if not has_any:
        output += "✅ No tracked TODOs found.\n\n"

    # 4. Footer with UTC timestamp
    now_utc = datetime.now(timezone.utc).strftime("%Y-%m-%d %H:%M:%S UTC")
    output += "---\n</br>"
    output += f"*Generated by **TODO Debt Tracker** action at {now_utc}*"

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

    # Print to stdout for the "gh issue edit" command
    print(report)


if __name__ == "__main__":
    main()
