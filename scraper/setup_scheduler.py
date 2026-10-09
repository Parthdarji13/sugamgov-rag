"""
scraper/setup_scheduler.py
===========================
Sets up Windows Task Scheduler to run the daily scraper automatically.

Run this ONCE to register the scheduled task:
  python scraper/setup_scheduler.py

To remove the task:
  python scraper/setup_scheduler.py --remove

What it creates:
  - Task Name: SugamGov Daily Scheme Scraper
  - Trigger:   Daily at 06:00 AM
  - Action:    python -m scraper.run_daily (from the project root)
  - Log:       scraper/logs/cron.log

Requirements:
  - Must be run as Administrator (or it will ask for elevation)
  - Python must be in PATH (uses venv python if available)
"""

import os
import sys
import subprocess
import argparse
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
TASK_NAME = "SugamGov Daily Scheme Scraper"
LOG_FILE = PROJECT_ROOT / "scraper" / "logs" / "cron.log"


def get_python_exe() -> str:
    """Returns the Python executable path (prefers venv)."""
    venv_python = PROJECT_ROOT / "venv" / "Scripts" / "python.exe"
    if venv_python.exists():
        return str(venv_python)
    return sys.executable


def create_task(run_hour: int = 6, run_minute: int = 0) -> None:
    """Registers the Windows Task Scheduler task."""
    python_exe = get_python_exe()
    LOG_FILE.parent.mkdir(parents=True, exist_ok=True)

    # Build the command that will run daily
    cmd_action = (
        f'"{python_exe}" -m scraper.run_daily '
        f'>> "{LOG_FILE}" 2>&1'
    )

    # schtasks XML is the most reliable way to create tasks with all options
    task_xml = f"""<?xml version="1.0" encoding="UTF-16"?>
<Task version="1.2" xmlns="http://schemas.microsoft.com/windows/2004/02/mit/task">
  <Triggers>
    <CalendarTrigger>
      <StartBoundary>2024-01-01T{run_hour:02d}:{run_minute:02d}:00</StartBoundary>
      <ScheduleByDay>
        <DaysInterval>1</DaysInterval>
      </ScheduleByDay>
    </CalendarTrigger>
  </Triggers>
  <Actions Context="Author">
    <Exec>
      <Command>{python_exe}</Command>
      <Arguments>-m scraper.run_daily</Arguments>
      <WorkingDirectory>{PROJECT_ROOT}</WorkingDirectory>
    </Exec>
  </Actions>
  <Settings>
    <DisallowStartIfOnBatteries>false</DisallowStartIfOnBatteries>
    <StopIfGoingOnBatteries>false</StopIfGoingOnBatteries>
    <ExecutionTimeLimit>PT2H</ExecutionTimeLimit>
    <MultipleInstancesPolicy>IgnoreNew</MultipleInstancesPolicy>
  </Settings>
  <Principals>
    <Principal id="Author">
      <LogonType>InteractiveToken</LogonType>
      <RunLevel>HighestAvailable</RunLevel>
    </Principal>
  </Principals>
</Task>"""

    xml_path = PROJECT_ROOT / "scraper" / "logs" / "task_definition.xml"
    xml_path.write_text(task_xml, encoding="utf-16")

    # Register with schtasks
    result = subprocess.run(
        ["schtasks", "/Create", "/F", "/XML", str(xml_path), "/TN", TASK_NAME],
        capture_output=True, text=True,
    )

    if result.returncode == 0:
        print(f"✅ Task '{TASK_NAME}' scheduled successfully.")
        print(f"   Python: {python_exe}")
        print(f"   Working dir: {PROJECT_ROOT}")
        print(f"   Runs: Daily at {run_hour:02d}:{run_minute:02d} AM")
        print(f"   Log file: {LOG_FILE}")
        print()
        print("To verify: Open Task Scheduler → Task Scheduler Library → look for 'SugamGov Daily Scheme Scraper'")
        print("To run NOW for testing: python -m scraper.run_daily --dry-run --max-pages 2")
    else:
        print(f"❌ Failed to create task:")
        print(result.stderr)
        print()
        print("Try running this script as Administrator.")
        print()
        print("OR run the scraper manually with:")
        print(f"  cd {PROJECT_ROOT}")
        print("  python -m scraper.run_daily")


def remove_task() -> None:
    """Removes the Windows Task Scheduler task."""
    result = subprocess.run(
        ["schtasks", "/Delete", "/F", "/TN", TASK_NAME],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(f"✅ Task '{TASK_NAME}' removed successfully.")
    else:
        print(f"Task not found or already removed: {result.stderr.strip()}")


def show_status() -> None:
    """Shows current task status."""
    result = subprocess.run(
        ["schtasks", "/Query", "/TN", TASK_NAME, "/FO", "LIST"],
        capture_output=True, text=True,
    )
    if result.returncode == 0:
        print(result.stdout)
    else:
        print(f"Task '{TASK_NAME}' not found. Run setup_scheduler.py to create it.")


if __name__ == "__main__":
    p = argparse.ArgumentParser(description="SugamGov Scheduler Setup")
    p.add_argument("--remove", action="store_true", help="Remove the scheduled task")
    p.add_argument("--status", action="store_true", help="Show task status")
    p.add_argument("--hour",   type=int, default=6,  help="Run hour (24h, default: 6 = 6 AM)")
    p.add_argument("--minute", type=int, default=0,  help="Run minute (default: 0)")
    args = p.parse_args()

    if args.remove:
        remove_task()
    elif args.status:
        show_status()
    else:
        create_task(run_hour=args.hour, run_minute=args.minute)
