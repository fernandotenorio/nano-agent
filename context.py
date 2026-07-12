# context.py

import logging
from pathlib import Path
from config import AppConfig

def gather_context_files(app_config: AppConfig, root: Path, cwd: Path) -> str:
    """
    Gathers context from AGENTS.md files along the lineage from root down to cwd.
    Global first, then root, down to cwd.
    """
    if not cwd.is_relative_to(root):
        raise ValueError(f"Current directory '{cwd}' is not within the specified root '{root}'.")

    files_to_check: list[Path] = []

    # 1. Global file
    global_agents = app_config.home_config_dir / "AGENTS.md"
    files_to_check.append(global_agents)

    # 2. Walk down from root to cwd
    rel_path = cwd.relative_to(root)
    
    current_dir = root
    files_to_check.append(current_dir / "AGENTS.md")
    
    for part in rel_path.parts:
        current_dir = current_dir / part
        files_to_check.append(current_dir / "AGENTS.md")

    # 3. Read and concatenate
    concatenated_texts: list[str] = []
    
    for file_path in files_to_check:
        if file_path.exists() and file_path.is_file():
            try:
                content = file_path.read_text(encoding="utf-8").strip()
                if content:
                    concatenated_texts.append(f"--- From {file_path} ---\n{content}")
            except Exception as e:
                logging.warning("Failed to read context file %s: %s", file_path, e)

    return "\n\n".join(concatenated_texts)