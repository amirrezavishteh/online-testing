#!/usr/bin/env python3
"""
Transfer fine-tuned SLM model to remote SSH server.

Usage:
    python scripts/transfer_model.py \\
      --local-model-dir /path/to/bait-slm-v1 \\
      --remote-user amirreza_vishteh \\
      --remote-host dmla100 \\
      --remote-base-dir /media/external20/amirreza_vishteh
"""

import argparse
import subprocess
import sys
from pathlib import Path
from typing import Optional

from loguru import logger


def check_ssh_connectivity(remote_user: str, remote_host: str) -> bool:
    """Check if SSH connection is available."""
    try:
        result = subprocess.run(
            ["ssh", "-q", f"{remote_user}@{remote_host}", "echo 'OK'"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"SSH check failed: {e}")
        return False


def create_remote_dir(remote_user: str, remote_host: str, remote_path: str) -> bool:
    """Create remote directory via SSH."""
    try:
        result = subprocess.run(
            ["ssh", f"{remote_user}@{remote_host}", f"mkdir -p '{remote_path}'"],
            capture_output=True,
            timeout=10
        )
        return result.returncode == 0
    except Exception as e:
        logger.error(f"Failed to create remote directory: {e}")
        return False


def get_remote_disk_space(
    remote_user: str,
    remote_host: str,
    remote_base_dir: str
) -> Optional[float]:
    """Get available disk space on remote in GB."""
    try:
        result = subprocess.run(
            ["ssh", f"{remote_user}@{remote_host}",
             f"df '{remote_base_dir}' | tail -1 | awk '{{print $4}}'"],
            capture_output=True,
            timeout=10,
            text=True
        )
        if result.returncode == 0:
            space_kb = int(result.stdout.strip())
            return space_kb / 1024 / 1024  # Convert to GB
        return None
    except Exception as e:
        logger.warning(f"Could not check remote disk space: {e}")
        return None


def verify_remote_model(remote_user: str, remote_host: str, remote_path: str) -> bool:
    """Verify that config.json exists on remote."""
    try:
        result = subprocess.run(
            ["ssh", f"{remote_user}@{remote_host}",
             f"test -f '{remote_path}/config.json'"],
            capture_output=True,
            timeout=5
        )
        return result.returncode == 0
    except Exception as e:
        logger.warning(f"Could not verify remote model: {e}")
        return False


def transfer_with_rsync(
    local_model_dir: str,
    remote_user: str,
    remote_host: str,
    remote_model_dir: str
) -> bool:
    """Transfer using rsync (preferred method)."""
    logger.info("Transferring model with rsync...")

    try:
        result = subprocess.run([
            "rsync", "-avz", "--progress",
            "--exclude=*.pyc",
            "--exclude=__pycache__",
            "--exclude=.git",
            f"{local_model_dir}/",
            f"{remote_user}@{remote_host}:{remote_model_dir}/"
        ], timeout=3600)  # 1 hour timeout

        return result.returncode == 0

    except FileNotFoundError:
        logger.warning("rsync not found, will try scp instead")
        return False
    except subprocess.TimeoutExpired:
        logger.error("rsync timeout (transfer took > 1 hour)")
        return False
    except Exception as e:
        logger.error(f"rsync transfer failed: {e}")
        return False


def transfer_with_scp(
    local_model_dir: str,
    remote_user: str,
    remote_host: str,
    remote_model_dir: str
) -> bool:
    """Transfer using scp (fallback method)."""
    logger.info("Transferring model with scp...")

    try:
        local_path = Path(local_model_dir)
        files = list(local_path.glob("**/*"))

        # Count files for progress
        files = [f for f in files if f.is_file()]
        total_files = len(files)

        logger.info(f"Transferring {total_files} files...")

        for idx, file in enumerate(files, 1):
            relative_path = file.relative_to(local_path)
            remote_file_path = f"{remote_user}@{remote_host}:{remote_model_dir}/{relative_path}"

            # Create parent directory on remote
            parent_dir = str(relative_path.parent)
            if parent_dir != ".":
                subprocess.run(
                    ["ssh", f"{remote_user}@{remote_host}",
                     f"mkdir -p '{remote_model_dir}/{parent_dir}'"],
                    capture_output=True,
                    timeout=5
                )

            # Transfer file
            result = subprocess.run(
                ["scp", str(file), remote_file_path],
                capture_output=True,
                timeout=300  # 5 minute per file
            )

            if result.returncode != 0:
                logger.error(f"Failed to transfer {file}")
                return False

            if idx % max(1, total_files // 10) == 0:
                logger.info(f"Progress: {idx}/{total_files} files")

        return True

    except FileNotFoundError:
        logger.error("scp not found")
        return False
    except Exception as e:
        logger.error(f"scp transfer failed: {e}")
        return False


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description="Transfer fine-tuned SLM model to remote SSH server"
    )
    parser.add_argument("--local-model-dir", required=True,
                        help="Local model directory path")
    parser.add_argument("--remote-user", required=True,
                        help="Remote SSH username")
    parser.add_argument("--remote-host", required=True,
                        help="Remote SSH hostname")
    parser.add_argument("--remote-base-dir", required=True,
                        help="Remote base directory")
    parser.add_argument("--remote-folder-name", default="bait-slm-v1",
                        help="Remote folder name for the model")

    args = parser.parse_args()

    # Validate local path
    local_path = Path(args.local_model_dir)
    if not local_path.exists():
        logger.error(f"Local model directory not found: {args.local_model_dir}")
        sys.exit(1)

    if not (local_path / "config.json").exists():
        logger.error(f"config.json not found in {args.local_model_dir}")
        sys.exit(1)

    local_size = sum(f.stat().st_size for f in local_path.rglob("*") if f.is_file())
    local_size_gb = local_size / 1e9

    remote_model_dir = f"{args.remote_base_dir}/{args.remote_folder_name}"

    # Print summary
    print("\n" + "="*70)
    print("MODEL TRANSFER TO REMOTE SERVER")
    print("="*70)
    print(f"Local model:    {args.local_model_dir}")
    print(f"Size:           {local_size_gb:.2f} GB")
    print(f"Remote server:  {args.remote_user}@{args.remote_host}")
    print(f"Remote path:    {remote_model_dir}")
    print("="*70 + "\n")

    # Test SSH connection
    logger.info("[1] Testing SSH connection...")
    if not check_ssh_connectivity(args.remote_user, args.remote_host):
        logger.error(f"Cannot connect to {args.remote_user}@{args.remote_host}")
        logger.error("Make sure SSH key is configured and host is reachable")
        sys.exit(1)
    logger.info("✓ SSH connection OK\n")

    # Create remote directory
    logger.info("[2] Creating remote directory...")
    if not create_remote_dir(args.remote_user, args.remote_host, remote_model_dir):
        logger.error("Failed to create remote directory")
        sys.exit(1)
    logger.info(f"✓ Created: {remote_model_dir}\n")

    # Check disk space
    logger.info("[3] Checking remote disk space...")
    remote_space_gb = get_remote_disk_space(
        args.remote_user,
        args.remote_host,
        args.remote_base_dir
    )
    if remote_space_gb:
        logger.info(f"Available: {remote_space_gb:.2f} GB")
        if remote_space_gb < local_size_gb + 5:
            logger.warning("Not enough space on remote! Consider freeing space.")
            sys.exit(1)
    logger.info()

    # Transfer
    logger.info("[4] Transferring model...")
    success = transfer_with_rsync(
        args.local_model_dir,
        args.remote_user,
        args.remote_host,
        remote_model_dir
    )

    if not success:
        logger.info("rsync unavailable, trying scp...")
        success = transfer_with_scp(
            args.local_model_dir,
            args.remote_user,
            args.remote_host,
            remote_model_dir
        )

    if not success:
        logger.error("Transfer failed")
        sys.exit(1)

    logger.info()

    # Verify
    logger.info("[5] Verifying transfer...")
    if verify_remote_model(args.remote_user, args.remote_host, remote_model_dir):
        logger.info("✓ Remote model verified\n")
    else:
        logger.warning("Could not verify remote model\n")

    # Success
    print("="*70)
    print("✓ TRANSFER COMPLETE")
    print("="*70)
    print(f"Remote model location: {remote_model_dir}\n")
    print("To use this model on the server:")
    print(f"  python detection_server.py --slm-repo-id {remote_model_dir} ...\n")


if __name__ == "__main__":
    main()
