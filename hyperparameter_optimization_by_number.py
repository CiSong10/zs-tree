from pathlib import Path
import geopandas as gpd
import pandas as pd
import optuna
from optuna.samplers import TPESampler
from pathlib import Path

from utility.zs_utility import clean_crowns, filter_by_shape
from pipeline import process_tiles

# ======= Configs ========
# Search bounds
DIAM_LOW, DIAM_HIGH = 10, 120  # integers
FLOW_LOW, FLOW_HIGH = 0.3, 1.5 
CELL_LOW, CELL_HIGH = -25, -9  # int

# Optuna settings
N_TRIALS    = 50


def main():
    # tiles_list = [21, 26, 27, 31, 32, 33, 36, 37]
    # for tile_num in tiles_list:
        # print(f"\n{tile_num}")
    run_trials(tile_num=39)


def objective(trial: optuna.Trial, tile_num) -> float:
    """
    Optuna objective. 
    Returns negative F1 so Optuna minimises -> maximises F1
    """
    img_dir = Path("imagery")
    tif_path   = img_dir / f"Coshocton_{str(tile_num)}.tif"
    mask_path  = img_dir / f"Coshocton_mask_{str(tile_num)}.tif"
    output_gdb = Path("output") / f"Coshocton_{str(tile_num)}_trials.gdb"

    diameter       = trial.suggest_int("diameter", DIAM_LOW, DIAM_HIGH)
    cell_threshold = trial.suggest_int("cellprob_threshold", CELL_LOW, CELL_HIGH)
    flow_threshold = trial.suggest_float("flow_threshold", FLOW_LOW, FLOW_HIGH)

    # 1. Generate predictions
    try:
        tree_crowns = process_tiles(tif_path, mask_path, diameter, flow_threshold, cell_threshold, ksize=1)
        tree_crowns = filter_by_shape(tree_crowns)
        pred = clean_crowns(tree_crowns, area_threshold=20, verbose=False)
    except (RuntimeError, TypeError) as e:
        print(e)
        return 0

    # 2. persist to GDB
    layer_name = f"trial{trial.number}_d{diameter}_c{abs(cell_threshold)}"
    pred.to_file(output_gdb, driver="OpenFileGDB", layer=layer_name,
                 layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"},)

    # 3
    count = pred.shape[0]

    return -count  # Optuna minimises, so negate


def run_trials(tile_num):
    output_gdb = Path("output") / f"Coshocton_{str(tile_num)}_trials.gdb"
    final_gdb  = Path("output") / f"Coshocton_tiles.gdb"

    sampler = TPESampler() #(seed=42)   # seed for reproducibility
    study = optuna.create_study(sampler=sampler, direction="minimize", study_name=f"{str(tile_num)}")
    
    # Warm-start: a few hand-picked trials so TPE has something to learn from
    # before it starts proposing. Remove if you want a fully blind search.
    study.enqueue_trial({"diameter": 30, "cellprob_threshold": -10, "flow_threshold": 1.2})
    study.enqueue_trial({"diameter": 50, "cellprob_threshold": -11, "flow_threshold": 0.5})
    study.enqueue_trial({"diameter": 80, "cellprob_threshold": -13, "flow_threshold": 1})
    
    study.optimize(lambda trial: objective(trial, tile_num), n_trials=N_TRIALS)

    # ======= Results ========================================================
    best = study.best_trial
    print("\n========== Best trial ==========")
    print(f" diameter           : {best.params['diameter']}")
    print(f" cellprob_threshold : {best.params['cellprob_threshold']}")
    print(f" flow_threshold     : {best.params['flow_threshold']:.1f}")
    print(f" Count              : {best.value}")

    # ======= Persist best trial layer to final GDB ========================
    diameter       = best.params["diameter"]
    cell_threshold = best.params["cellprob_threshold"]

    best_layer_name = f"trial{best.number}_d{diameter}_c{abs(cell_threshold)}"
    best_crowns = gpd.read_file(output_gdb, layer=best_layer_name)

    final_layer_name = f"tile{str(tile_num)}_d{diameter}_c{abs(cell_threshold)}"
    best_crowns.to_file(final_gdb, driver="OpenFileGDB", layer=final_layer_name,
                         layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"})


if __name__ == "__main__":
    main()
