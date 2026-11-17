"""Resolve whether a mapped feature has a visible presence at street level."""
import re


def below_surface(tags):
    """An underground footprint or tunnel must not occupy the surface carriageway."""
    if tags.get("railway") == "subway_entrance" or tags.get("tunnel") == "building_passage":
        return False
    if tags.get("location") in {"underground", "underwater"}:
        return True
    if tags.get("tunnel") in {"yes", "culvert", "flooded"}:
        return True
    levels = re.split(r"[;,]", tags.get("level", ""))
    try:
        if levels and all(float(level) < 0 for level in levels):
            return True
    except ValueError:
        pass
    return tags.get("parking") == "underground" and tags.get("amenity") == "parking"
