import sys

import psutil


def get_username_by_pid(pid):
    try:
        process = psutil.Process(pid)
        username = process.username()
        return username
    except psutil.NoSuchProcess:
        return None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python whois.py <pid>")
        sys.exit(1)

    pid = int(sys.argv[1])
    username = get_username_by_pid(pid)
    if username:
        print(f"Username for PID {pid}: {username}")
    else:
        print(f"PID {pid} not found or no username available.")
