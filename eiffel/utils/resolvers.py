"""OmegaConf resolvers for Eiffel."""

from logging import getLogger
from pathlib import Path

from omegaconf import OmegaConf

log = getLogger(__name__)


def get_git_root():
    """OmegaConf resolver to get the git directory."""
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / ".git").exists():
            return str(p)
    log.warning("No git directory found.")
    return str(Path.cwd())

def get_eiffel_anchor():
    """OmegaConf resolver to get the Eiffel anchor directory.
    
    The anchor directory is the directory containing either a .eiffel-anchor file, or
    the root of the parent git repository.
    """
    for p in (Path.cwd(), *Path.cwd().parents):
        if (p / ".eiffel-anchor").exists():
            return str(p)
        if (p / ".git").exists():
            return str(p)
    log.error("No anchor found.")
