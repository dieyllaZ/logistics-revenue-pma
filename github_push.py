"""
github_push.py
Run: python github_push.py
"""
import subprocess
import sys
import getpass

# ── Fill in these two (no token here) ────────────────────────
GITHUB_USERNAME = "dieyllaZ"
REPO_NAME       = "logistics-revenue-pma"
COMMIT_MESSAGE  = "Sprint 1 Q1: EDA, data quality checks, cleaning pipeline"
# ─────────────────────────────────────────────────────────────


def run(command, description=""):
    print(f"\n>>> {description or command}")
    result = subprocess.run(command, shell=True,
                            capture_output=True, text=True)
    if result.stdout:
        print(result.stdout.strip())
    if result.stderr:
        print(result.stderr.strip())
    return result.returncode


def push_to_github():
    print("=" * 60)
    print("GITHUB PUSH SCRIPT")
    print("=" * 60)

    # Token entered at runtime — never stored in the file
    print("\nPaste your GitHub Personal Access Token below.")
    print("(It will not be shown on screen as you type)\n")
    GITHUB_TOKEN = getpass.getpass("GitHub Token: ")

    if not GITHUB_TOKEN.startswith("ghp_"):
        print("ERROR: Token does not look right. It should start with ghp_")
        sys.exit(1)

    run("git --version", "Checking git is installed")
    run(f'git config --global user.name "{GITHUB_USERNAME}"', "Setting username")
    run(f'git config --global user.email "{GITHUB_USERNAME}@users.noreply.github.com"',
        "Setting email")
    run("git init", "Initialising git repository")
    run("git add .", "Staging all files")
    run(f'git commit -m "{COMMIT_MESSAGE}"', "Creating commit")

    remote_url = (f"https://{GITHUB_TOKEN}@github.com/"
                  f"{GITHUB_USERNAME}/{REPO_NAME}.git")
    run("git remote remove origin", "Removing old remote")
    run(f"git remote add origin {remote_url}", "Adding remote")
    run("git branch -M main", "Setting branch to main")
    code = run("git push -u origin main", "Pushing to GitHub")

    print("\n" + "=" * 60)
    if code == 0:
        print("SUCCESS!")
        print(f"View at: https://github.com/{GITHUB_USERNAME}/{REPO_NAME}")
    else:
        print("PUSH FAILED — check the error above.")
    print("=" * 60)


if __name__ == "__main__":
    push_to_github()