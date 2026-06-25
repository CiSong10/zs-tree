from pathlib import Path
import geopandas as gpd
import pandas as pd
import optuna
from optuna.samplers import TPESampler

from zs_utility import clean_crowns, filter_by_shape
from pipeline import process_tiles
from cross_model.evaluate_iou import evaluate_tree_crowns

# ======= Configs ========

data_dir    = Path("/home/cisong/detectree2-implementation/data/brooklyn_1/")
tif_path    = data_dir / "ms" / "Brooklyn_train_1_utm.tif"
mask_path   = "imagery/Brooklyn1_CanopyMask.tif"
gt_path     = data_dir / "crowns" / "brooklyn_train_1.gpkg"
output_gdb  = "Brooklyn_zs.gdb"
results_csv = Path("output") / "optuna_results.csv"

# Search bounds
DIAMETER_LOW,  DIAMETER_HIGH  = 100, 250   # integers
THRESHOLD_LOW, THRESHOLD_HIGH = -4.0, 0.0  # floats

# Optuna settings
N_TRIALS    = 40   # increase for a finer search; 40 is a good starting point
IOU_THRESH  = 0.5  # matching threshold passed to evaluate_tree_crowns

gt_crowns = gpd.read_file(gt_path)

# ======= Optuna objective ===============================================

def objective(trial: optuna.Trial) -> float:
    """
    Returns negative F1 so Optuna minimises -> maximises F1
    """

    diameter  = trial.suggest_int("diameter", DIAMETER_LOW, DIAMETER_HIGH)
    threshold = trial.suggest_float("cellprob_threshold", THRESHOLD_LOW, THRESHOLD_HIGH)

    # 1. Generate predictions
    tree_crowns = process_tiles(tif_path, mask_path, diameter, threshold)
    tree_crowns = filter_by_shape(tree_crowns)
    pred = clean_crowns(tree_crowns, area_threshold=20, verbose=False)

    # 2. persist to GDB
    layer_name = f"Brooklyn_d{diameter}_t{abs(threshold):.1f}_trial{trial.number}"
    pred.to_file(output_gdb, driver="OpenFileGDB", layer=layer_name,
                 layer_options={"TARGET_ARCGIS_VERSION": "ARCGIS_PRO_3_2_OR_LATER"},)

    # 3. Evaluate against ground truth
    metrics = evaluate_tree_crowns(pred, gt_crowns, iou_threshold=IOU_THRESH)
    
    for k, v in metrics.items():
        trial.set_user_attr(k, v)
    trial.set_user_attr("layer_name", layer_name)

    f1 = metrics["F1"]
 
    return -f1  # Optuna minimises, so negate


def main():
    # ======= Run study ======================================================

    sampler = TPESampler(seed=42)   # seed for reproducibility
    study = optuna.create_study(sampler=sampler, direction="minimize")
    
    # Warm-start: a few hand-picked trials so TPE has something to learn from
    # before it starts proposing. Remove if you want a fully blind search.
    study.enqueue_trial({"diameter": 150, "cellprob_threshold": -2.0})
    study.enqueue_trial({"diameter": 120, "cellprob_threshold": -1.0})
    study.enqueue_trial({"diameter": 200, "cellprob_threshold": -3.0})
    
    study.optimize(objective, n_trials=N_TRIALS)


    # ======= Results ========================================================
    
    best = study.best_trial
    print("\n========== Best trial ==========")
    print(f" diameter           : {best.params['diameter']}")
    print(f" cellprob_threshold : {best.params['cellprob_threshold']:.3f}")
    print(f" F1                 : {-best.value:.4f}")
    print(f" mIoU               : {best.user_attrs['mIoU']:.4f}")
    print(f" Precision          : {best.user_attrs['Precision']:.4f}")
    print(f" Recall             : {best.user_attrs['Recall']:.4f}")
    print(f" GDB layer          : {best.user_attrs['layer_name']}")
    
    # Save full results table to CSV for plotting / auditing
    records = []
    for t in study.trials:
        records.append({
            "trial":               t.number,
            "diameter":            t.params["diameter"],
            "cellprob_threshold":  t.params["cellprob_threshold"],
            "F1":                  -t.value,
            **{k: v for k, v in t.user_attrs.items()},
        })
    
    df = pd.DataFrame(records).sort_values("F1", ascending=False)
    df.to_csv(results_csv, index=False)


if __name__ == "__main__":
    main()
