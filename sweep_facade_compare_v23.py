"""Facade coverage comparison, memory-light: per building, facade-layer triangle count and the height up to which
facade detail reaches, reference vs candidate. Usage: sweep_facade_compare_v23.py <reference.glb> <candidate.glb>"""
import sys
sys.argv=[sys.argv[0]]+sys.argv[1:]
from sweep_mass_compare_v23 import load  # noqa: E402  (reuses the light loader)
