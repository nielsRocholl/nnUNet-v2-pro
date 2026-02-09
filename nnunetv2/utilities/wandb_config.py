import os
from typing import Optional, List


def resolve_wandb_config(
    use_wandb: Optional[bool] = None,
    wandb_project: Optional[str] = None,
    wandb_entity: Optional[str] = None,
    wandb_run_name: Optional[str] = None,
    wandb_tags: Optional[List[str]] = None
) -> dict:
    """Resolve wandb config: args → env vars → defaults"""
    return {
        'use_wandb': use_wandb if use_wandb is not None else os.environ.get('nnUNet_use_wandb', '').lower() in ('true', '1', 't', 'yes'),
        'project': wandb_project or os.environ.get('nnUNet_wandb_project'),
        'entity': wandb_entity or os.environ.get('nnUNet_wandb_entity'),
        'run_name': wandb_run_name or os.environ.get('nnUNet_wandb_run_name'),
        'tags': wandb_tags or (os.environ.get('nnUNet_wandb_tags', '').split(',') if os.environ.get('nnUNet_wandb_tags') else None)
    }
