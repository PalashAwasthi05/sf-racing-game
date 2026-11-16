from pipeline.panel_finish import panel_recipe
from pipeline.material_recipes import RECIPES
def test_observed_bare_finish_beats_paint_prior():
    assert panel_recipe('#eeeae2',coating='bare')=='panel_silver'
    assert panel_recipe('#eeeae2')=='panel_coated'
    assert panel_recipe('#939a9d')=='panel_silver'
    assert panel_recipe('#bd3528')=='panel_coated'
    assert RECIPES['panel_coated']['metal']==0
