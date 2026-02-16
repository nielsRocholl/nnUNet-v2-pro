"""
Retired implementation of plan_experiments() with original print statements.
Preserved for reference.
"""

from typing import List, Optional, Tuple


def plan_experiments_original(dataset_ids: List[int], experiment_planner_class_name: str = 'ExperimentPlanner',
                     gpu_memory_target_in_gb: float = None, preprocess_class_name: str = 'DefaultPreprocessor',
                     overwrite_target_spacing: Optional[Tuple[float, ...]] = None,
                     overwrite_plans_name: Optional[str] = None):
    """
    overwrite_target_spacing ONLY applies to 3d_fullres and 3d_cascade fullres!
    """
    if experiment_planner_class_name == 'ExperimentPlanner':
        print("\n############################\n"
              "INFO: You are using the old nnU-Net default planner. We have updated our recommendations. "
              "Please consider using those instead! "
              "Read more here: https://github.com/MIC-DKFZ/nnUNet/blob/master/documentation/resenc_presets.md"
              "\n############################\n")
    experiment_planner = recursive_find_python_class(join(nnunetv2.__path__[0], "experiment_planning"),
                                                     experiment_planner_class_name,
                                                     current_module="nnunetv2.experiment_planning")
    plans_identifier = None
    for d in dataset_ids:
        _, plans_identifier = plan_experiment_dataset(d, experiment_planner, gpu_memory_target_in_gb,
                                                      preprocess_class_name,
                                                      overwrite_target_spacing, overwrite_plans_name)
    return plans_identifier
