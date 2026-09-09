from dataclasses import dataclass,asdict

@dataclass(frozen=True)
class Policy:
 roof_cell_m:float=1.5
 roof_height_step_m:float=1.0
 minimum_roof_samples:int=4
 minimum_roof_component_cells:int=4
 ground_clearance_m:float=.30
 minimum_independent_views:int=2
 minimum_detail_confidence:float=.9
 maximum_observation_age_days:int=730
 max_cheap_attempts:int=2
 max_strong_attempts:int=1
 max_capture_budget:int=240
 minimum_visible_evidence_fraction:float=.95
 maximum_open_edges:int=0

def manifest():return asdict(Policy())
