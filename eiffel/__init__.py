"""Eiffel: Evaluation framework for FL-based intrusion detection using Flower."""

import logging
import os

os.environ["TF_CPP_MIN_LOG_LEVEL"] = "3"
os.environ["OPENBLAS_NUM_THREADS"] = "64"
os.environ["OMP_NUM_THREADS"] = "64"

from omegaconf import OmegaConf

from eiffel.utils.resolvers import get_git_root

OmegaConf.register_new_resolver("gitdir", get_git_root)
OmegaConf.register_new_resolver("if", lambda cond, a, b: a if cond else b)
OmegaConf.register_new_resolver("sanitize", lambda s: s.replace("/", "_"))
