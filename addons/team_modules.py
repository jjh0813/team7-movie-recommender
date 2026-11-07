"""Load the exact original modules and original integration without copying code."""
from pathlib import Path
import sys
import importlib.util
from memory_cf import cf_module
from movie_content import content_based

ROOT=Path(__file__).resolve().parents[1]
sys.modules.setdefault('cf_module',cf_module)
sys.modules.setdefault('content_based',content_based)

def load_combiner():
    spec=importlib.util.spec_from_file_location('team_original_combiner',ROOT/'code result/PerformCampare.py')
    mod=importlib.util.module_from_spec(spec);spec.loader.exec_module(mod)
    return mod
