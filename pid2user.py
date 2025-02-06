import argparse
from collections import defaultdict

def check_dependencies():
    missing = []
    try:
        import psutil
    except ImportError:
        missing.append("psutil")
    try:
        import tabulate
    except ImportError:
        missing.append("tabulate")

    if missing:
        print(f"Missing required packages: {', '.join(missing)}")
        print("Install with: pip install " + " ".join(missing))
        exit(1)
    return psutil, tabulate

def get_process_info(pid, psutil):
    try:
        proc = psutil.Process(pid)
        return [{
            'PID': pid,
            'User': proc.username(),
            'Command': ' '.join(proc.cmdline() or []),
            'CPU Memory (MB)': f"{proc.memory_info().rss / 1024 / 1024:.1f}"
        }]
    except (psutil.NoSuchProcess, psutil.AccessDenied):
        return []

def get_python_users(psutil):
    process_list = []
    for proc in psutil.process_iter(['pid', 'name', 'username', 'cmdline']):
        try:
            if 'python' in proc.info['name'].lower():
                process_list.append({
                    'PID': proc.info['pid'],
                    'User': proc.info['username'],
                    'Command': ' '.join(proc.info['cmdline'] or []),
                    'CPU Memory (MB)': f"{proc.memory_info().rss / 1024 / 1024:.1f}"
                })
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    return process_list

if __name__ == '__main__':
    psutil, tabulate = check_dependencies()

    parser = argparse.ArgumentParser(description='Monitor Python processes')
    parser.add_argument('--pid', type=int, help='Specific PID to monitor')
    args = parser.parse_args()

    if args.pid:
        processes = get_process_info(args.pid, psutil)
    else:
        processes = get_python_users(psutil)

    if processes:

        for p in processes:
            print(p)
    else:
        print("No Python processes found" if not args.pid else f"No Python process found with PID {args.pid}")
