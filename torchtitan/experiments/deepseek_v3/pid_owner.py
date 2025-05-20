import pwd
import sys

import psutil


def get_process_owner(pid):
    try:
        process = psutil.Process(pid)
        uid = process.uids().real
        username = pwd.getpwuid(uid).pw_name

        print(f"Process ID: {pid}")
        print(f"User ID (UID): {uid}")
        print(f"Username: {username}")

        return uid, username
    except psutil.NoSuchProcess:
        print(f"No process found with PID {pid}")
        return None, None
    except Exception as e:
        print(f"Error: {e}")
        return None, None


if __name__ == "__main__":
    if len(sys.argv) != 2:
        print("Usage: python script.py <pid>")
        sys.exit(1)

    try:
        pid = int(sys.argv[1])
        get_process_owner(pid)
    except ValueError:
        print("PID must be an integer")
        sys.exit(1)
