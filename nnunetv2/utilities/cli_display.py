from contextlib import contextmanager
from io import StringIO
import sys
from time import time
from typing import Optional, Dict, Any, List, Tuple

import numpy as np
from rich.console import Console
from rich.progress import Progress, SpinnerColumn, BarColumn, TextColumn, TimeElapsedColumn, TimeRemainingColumn
from rich.table import Table
from rich.panel import Panel
from rich import box
from rich.rule import Rule


class PlanningDisplay:
    def __init__(self, dataset_name: str, verbose: bool = False):
        self.dataset_name = dataset_name
        self.verbose = verbose
        self.console = Console()
        self.progress = None
        self.task_ids = {}
        self.start_time = None
        self.messages = []
        self.steps = [
            "Determining transpose",
            "Computing target spacing",
            "Planning 3D fullres configuration",
            "Planning 3D lowres configuration",
            "Saving plans"
        ]
        self.current_step = 0

    def __enter__(self):
        self.start_time = time()
        if not self.verbose:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            )
            self.progress.start()
            
            for i, step in enumerate(self.steps):
                self.task_ids[step] = self.progress.add_task(
                    f"Step {i+1}/{len(self.steps)}: {step}...",
                    total=100
                )
        
        self.console.print(Panel(
            f"[bold cyan]Planning Experiment: {self.dataset_name}[/bold cyan]",
            border_style="cyan"
        ))
        
        # Show planning pipeline flow
        self._show_pipeline_flow()
        
        return self
    
    def _show_pipeline_flow(self):
        """Display the experiment planning pipeline flow"""
        steps = ["Transpose", "Spacing", "Plan Fullres", "Plan Lowres", "Save"]
        flow_text = " → ".join([f"[cyan]{s}[/cyan]" for s in steps])
        self.console.print(f"\n[dim]Pipeline:[/dim] {flow_text}\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop()
        
        # Show captured messages if any
        if self.messages:
            self._show_messages()
        
        elapsed = time() - self.start_time if self.start_time else 0
        if exc_type is None:
            self.console.print(f"\n[bold green]✓ Planning completed in {elapsed:.1f}s[/bold green]")
            self._show_separator()
        return False
    
    def _show_separator(self):
        """Display a separator line between steps"""
        self.console.print(Rule(style="dim"))
    
    def add_message(self, message: str, message_type: str = "info"):
        """Capture a message to display at the end"""
        self.messages.append((message, message_type))
    
    def _show_messages(self):
        """Display captured messages at the bottom"""
        if not self.messages:
            return
        
        for msg, msg_type in self.messages:
            if msg_type == "info":
                self.console.print(f"[dim]{msg}[/dim]")
            elif msg_type == "success":
                self.console.print(f"[green]✓ {msg}[/green]")
            elif msg_type == "warning":
                self.console.print(f"[yellow]⚠ {msg}[/yellow]")
        self.messages.clear()

    def update_step(self, step_name: str, progress: int = 100):
        if not self.verbose and self.progress and step_name in self.task_ids:
            self.progress.update(self.task_ids[step_name], completed=progress)
        elif self.verbose:
            self.console.print(f"[dim]✓ {step_name}[/dim]")

    def complete_step(self, step_name: str):
        self.update_step(step_name, 100)

    def show_configuration(self, config_name: str, config: Dict[str, Any]):
        table = Table(title=f"Configuration: {config_name}", box=box.ROUNDED, show_header=False)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        if 'patch_size' in config:
            table.add_row("Patch Size", str(config['patch_size']))
        if 'batch_size' in config:
            table.add_row("Batch Size", str(config['batch_size']))
        if 'spacing' in config:
            spacing_str = ', '.join(f"{x:.3f}" for x in config['spacing'])
            table.add_row("Target Spacing", f"[{spacing_str}]")
        if 'median_image_shape_after_cropping' in config:
            table.add_row("Median Shape", str(config['median_image_shape_after_cropping']))
        if 'num_pool_per_axis' in config:
            table.add_row("Pooling Stages", str(config['num_pool_per_axis']))
        if 'unet_base_num_features' in config:
            table.add_row("Base Features", str(config['unet_base_num_features']))
        
        self.console.print(Panel(table, border_style="green"))

    def show_info(self, message: str, style: str = "yellow"):
        self.console.print(f"[{style}]{message}[/{style}]")

    def show_warning(self, message: str):
        """Capture warning to display at the end"""
        self.messages.append((message, "warning"))

    def show_lowres_iteration(self, iteration: int, spacing: List[float], patch_size: Tuple[int, ...], median_shape: List[float]):
        self.console.print(
            f"[dim]Iteration {iteration}: spacing={spacing}, "
            f"patch_size={patch_size}, median_shape={median_shape}[/dim]"
        )


class FingerprintDisplay:
    def __init__(self, dataset_name: str, num_cases: int, verbose: bool = False):
        self.dataset_name = dataset_name
        self.num_cases = num_cases
        self.verbose = verbose
        self.console = Console()
        self.progress = None
        self.task_ids = {}
        self.case_task_id = None
        self.start_time = None
        self.messages = []
        self.steps = [
            "Verifying dataset integrity",
            "Analyzing cases",
            "Aggregating results",
            "Computing statistics",
            "Saving fingerprint"
        ]

    def __enter__(self):
        self.start_time = time()
        if not self.verbose:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            )
            self.progress.start()
            
            for i, step in enumerate(self.steps):
                self.task_ids[step] = self.progress.add_task(
                    f"Step {i+1}/{len(self.steps)}: {step}...",
                    total=100
                )
            
            self.case_task_id = self.progress.add_task(
                f"Cases: 0/{self.num_cases}",
                total=self.num_cases
            )
        
        self.console.print(Panel(
            f"[bold blue]Extracting Fingerprint: {self.dataset_name}[/bold blue]",
            border_style="blue"
        ))
        
        # Show fingerprint extraction pipeline flow
        self._show_pipeline_flow()
        
        return self
    
    def _show_pipeline_flow(self):
        """Display the fingerprint extraction pipeline flow"""
        analyze_steps = ["Load", "Crop", "Collect Intensities"]
        analyze_flow = " → ".join([f"[cyan]{s}[/cyan]" for s in analyze_steps])
        compute_steps = ["Median Spacing", "Intensity Stats"]
        compute_flow = " → ".join([f"[cyan]{s}[/cyan]" for s in compute_steps])
        flow_text = f"[cyan]Verify Integrity[/cyan] → Analyze Cases ({analyze_flow}) → [cyan]Aggregate[/cyan] → Compute ({compute_flow}) → [cyan]Save[/cyan]"
        self.console.print(f"\n[dim]Pipeline:[/dim] {flow_text}\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop()
        
        # Show captured messages if any
        if self.messages:
            self._show_messages()
        
        elapsed = time() - self.start_time if self.start_time else 0
        if exc_type is None:
            self.console.print(f"\n[bold green]✓ Fingerprint extraction completed in {elapsed:.1f}s[/bold green]")
            self._show_separator()
        return False
    
    def _show_separator(self):
        """Display a separator line between steps"""
        self.console.print(Rule(style="dim"))
    
    def add_message(self, message: str, message_type: str = "info"):
        """Capture a message to display at the end"""
        self.messages.append((message, message_type))
    
    def _show_messages(self):
        """Display captured messages at the bottom"""
        if not self.messages:
            return
        
        for msg, msg_type in self.messages:
            if msg_type == "info":
                self.console.print(f"[dim]{msg}[/dim]")
            elif msg_type == "success":
                self.console.print(f"[green]✓ {msg}[/green]")
            elif msg_type == "warning":
                self.console.print(f"[yellow]⚠ {msg}[/yellow]")
        self.messages.clear()

    def update_step(self, step_name: str, progress: int = 100):
        if not self.verbose and self.progress and step_name in self.task_ids:
            self.progress.update(self.task_ids[step_name], completed=progress)
        elif self.verbose:
            self.console.print(f"[dim]✓ {step_name}[/dim]")

    def complete_step(self, step_name: str):
        self.update_step(step_name, 100)

    def update_cases(self, completed: int):
        if not self.verbose and self.progress and self.case_task_id is not None:
            self.progress.update(self.case_task_id, completed=completed, description=f"Cases: {completed}/{self.num_cases}")

    def show_summary(self, fingerprint: Dict[str, Any]):
        table = Table(title="Fingerprint Summary", box=box.ROUNDED, show_header=False)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        if 'spacings' in fingerprint:
            median_spacing = np.median([s for s in fingerprint['spacings']], axis=0)
            spacing_str = ', '.join(f"{x:.3f}" for x in median_spacing)
            table.add_row("Median Spacing", f"[{spacing_str}]")
        if 'shapes_after_crop' in fingerprint:
            median_shape = np.median([s for s in fingerprint['shapes_after_crop']], axis=0)
            table.add_row("Median Shape", str([int(x) for x in median_shape]))
        if 'median_relative_size_after_cropping' in fingerprint:
            table.add_row("Relative Size After Crop", f"{fingerprint['median_relative_size_after_cropping']:.2%}")
        
        self.console.print(Panel(table, border_style="blue"))


class PreprocessingDisplay:
    def __init__(self, dataset_name: str, configuration: str, num_cases: int, verbose: bool = False):
        self.dataset_name = dataset_name
        self.configuration = configuration
        self.num_cases = num_cases
        self.verbose = verbose
        self.console = Console()
        self.progress = None
        self.task_ids = {}
        self.case_task_id = None
        self.start_time = None
        self.messages = []
        self.steps = [
            "Loading plans",
            "Processing cases",
            "Copying ground truth"
        ]

    def __enter__(self):
        self.start_time = time()
        if not self.verbose:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            )
            self.progress.start()
            
            for i, step in enumerate(self.steps):
                self.task_ids[step] = self.progress.add_task(
                    f"Step {i+1}/{len(self.steps)}: {step}...",
                    total=100
                )
            
            self.case_task_id = self.progress.add_task(
                f"Cases: 0/{self.num_cases}",
                total=self.num_cases
            )
        
        self.console.print(Panel(
            f"[bold magenta]Preprocessing: {self.dataset_name} - {self.configuration}[/bold magenta]",
            border_style="magenta"
        ))
        
        # Show preprocessing pipeline flow
        self._show_pipeline_flow()
        
        return self
    
    def _show_pipeline_flow(self):
        """Display the preprocessing pipeline flow"""
        preprocess_steps = ["Transpose", "Crop", "Normalize", "Resample"]
        preprocess_flow = " → ".join([f"[cyan]{s}[/cyan]" for s in preprocess_steps])
        flow_text = f"[cyan]Load Plans[/cyan] → Process Cases ({preprocess_flow}) → [cyan]Copy GT[/cyan]"
        self.console.print(f"\n[dim]Pipeline:[/dim] {flow_text}\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop()
        
        # Show captured messages if any
        if self.messages:
            self._show_messages()
        
        elapsed = time() - self.start_time if self.start_time else 0
        if exc_type is None:
            self.console.print(f"\n[bold green]✓ Preprocessing completed in {elapsed:.1f}s[/bold green]")
            self._show_separator()
        return False
    
    def _show_separator(self):
        """Display a separator line between steps"""
        self.console.print(Rule(style="dim"))
    
    def add_message(self, message: str, message_type: str = "info"):
        """Capture a message to display at the end"""
        self.messages.append((message, message_type))
    
    def _show_messages(self):
        """Display captured messages at the bottom"""
        if not self.messages:
            return
        
        for msg, msg_type in self.messages:
            if msg_type == "info":
                self.console.print(f"[dim]{msg}[/dim]")
            elif msg_type == "success":
                self.console.print(f"[green]✓ {msg}[/green]")
            elif msg_type == "warning":
                self.console.print(f"[yellow]⚠ {msg}[/yellow]")
        self.messages.clear()

    def update_step(self, step_name: str, progress: int = 100):
        if not self.verbose and self.progress and step_name in self.task_ids:
            self.progress.update(self.task_ids[step_name], completed=progress)
        elif self.verbose:
            self.console.print(f"[dim]✓ {step_name}[/dim]")

    def complete_step(self, step_name: str):
        self.update_step(step_name, 100)

    def update_cases(self, completed: int):
        if not self.verbose and self.progress and self.case_task_id is not None:
            self.progress.update(self.case_task_id, completed=completed, description=f"Cases: {completed}/{self.num_cases}")


class TrainingDisplay:
    def __init__(self, dataset_name: str, configuration: str, fold: int, num_epochs: int, device: str, verbose: bool = False):
        self.dataset_name = dataset_name
        self.configuration = configuration
        self.fold = fold
        self.num_epochs = num_epochs
        self.device = device
        self.verbose = verbose
        self.console = Console()
        self.progress = None
        self.epoch_task_id = None
        self.train_batch_task_id = None
        self.val_batch_task_id = None
        self.start_time = None
        self.current_epoch = 0
        self.current_train_loss = None
        self.current_val_loss = None
        self.current_dice = None
        self.current_lr = None
        self.best_epoch = None
        self.best_dice = None
        self.num_train_batches = None
        self.num_val_batches = None

    def __enter__(self):
        self.start_time = time()
        if not self.verbose:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            )
            self.progress.start()
            
            self.epoch_task_id = self.progress.add_task(
                "Epoch 0/0",
                total=self.num_epochs
            )
            self.train_batch_task_id = self.progress.add_task(
                "Training...",
                total=100,
                visible=False
            )
            self.val_batch_task_id = self.progress.add_task(
                "Validation...",
                total=100,
                visible=False
            )
        
        device_str = "MPS (Apple Silicon GPU)" if self.device == "mps" else \
                     "CUDA (GPU)" if self.device == "cuda" else \
                     "CPU"
        
        self.console.print(Panel(
            f"[bold yellow]Training: {self.dataset_name} - {self.configuration} (Fold {self.fold})[/bold yellow]\n"
            f"Device: {device_str}",
            border_style="yellow"
        ))
        
        # Show training pipeline flow
        self._show_pipeline_flow()
        
        return self
    
    def _show_pipeline_flow(self):
        """Display the training pipeline flow"""
        train_steps = ["Train Batches", "Validate"]
        train_flow = " → ".join([f"[cyan]{s}[/cyan]" for s in train_steps])
        flow_text = f"[cyan]Initialize[/cyan] → [yellow]Train Epochs[/yellow] ({train_flow}) → [cyan]Save Checkpoint[/cyan]"
        self.console.print(f"\n[dim]Pipeline:[/dim] {flow_text}\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop()
        
        elapsed = time() - self.start_time if self.start_time else 0
        if exc_type is None:
            hours = int(elapsed // 3600)
            minutes = int((elapsed % 3600) // 60)
            seconds = int(elapsed % 60)
            if hours > 0:
                time_str = f"{hours}h {minutes}m {seconds}s"
            elif minutes > 0:
                time_str = f"{minutes}m {seconds}s"
            else:
                time_str = f"{seconds}s"
            self.console.print(f"\n[bold green]✓ Training completed in {time_str}[/bold green]")
        return False

    def show_configuration(self, config: Dict[str, Any]):
        table = Table(title="Training Configuration", box=box.ROUNDED, show_header=False)
        table.add_column("Property", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        if 'batch_size' in config:
            table.add_row("Batch Size", str(config['batch_size']))
        if 'patch_size' in config:
            table.add_row("Patch Size", str(config['patch_size']))
        if 'spacing' in config:
            spacing_str = ', '.join(f"{x:.3f}" for x in config['spacing'])
            table.add_row("Spacing", f"[{spacing_str}]")
        
        self.console.print(Panel(table, border_style="yellow"))

    def update_epoch(self, epoch: int, total_epochs: int = None):
        self.current_epoch = epoch
        if total_epochs is None:
            total_epochs = self.num_epochs
        
        if not self.verbose and self.progress and self.epoch_task_id is not None:
            self.progress.update(
                self.epoch_task_id,
                completed=epoch,
                description=f"Epoch {epoch}/{total_epochs}"
            )
        elif self.verbose:
            self.console.print(f"[dim]Epoch {epoch}/{total_epochs}[/dim]")

    def set_batch_counts(self, num_train_batches: int, num_val_batches: int):
        self.num_train_batches = num_train_batches
        self.num_val_batches = num_val_batches

    def update_training_batch(self, batch: int, total_batches: int = None, loss: float = None):
        if total_batches is None:
            total_batches = self.num_train_batches
        
        if not self.verbose and self.progress:
            if self.train_batch_task_id is not None:
                if not self.progress.tasks[self.train_batch_task_id].visible:
                    self.progress.update(self.train_batch_task_id, visible=True)
                    if self.val_batch_task_id is not None:
                        self.progress.update(self.val_batch_task_id, visible=False)
                
                self.progress.update(
                    self.train_batch_task_id,
                    completed=batch,
                    total=total_batches,
                    description=f"Training... Batch {batch}/{total_batches}" + (f" Loss: {loss:.4f}" if loss is not None else "")
                )
        elif self.verbose and loss is not None:
            self.console.print(f"[dim]  Training batch {batch}/{total_batches}, Loss: {loss:.4f}[/dim]")

    def update_validation_batch(self, batch: int, total_batches: int = None):
        if total_batches is None:
            total_batches = self.num_val_batches
        
        if not self.verbose and self.progress:
            if self.val_batch_task_id is not None:
                if not self.progress.tasks[self.val_batch_task_id].visible:
                    self.progress.update(self.val_batch_task_id, visible=True)
                    if self.train_batch_task_id is not None:
                        self.progress.update(self.train_batch_task_id, visible=False)
                
                self.progress.update(
                    self.val_batch_task_id,
                    completed=batch,
                    total=total_batches,
                    description=f"Validation... Batch {batch}/{total_batches}"
                )
        elif self.verbose:
            self.console.print(f"[dim]  Validation batch {batch}/{total_batches}[/dim]")

    def show_epoch_summary(self, train_loss: float, val_loss: float, dice: float, lr: float, epoch_time: float):
        self.current_train_loss = train_loss
        self.current_val_loss = val_loss
        self.current_dice = dice
        self.current_lr = lr
        
        table = Table(title=f"Epoch {self.current_epoch} Summary", box=box.ROUNDED, show_header=False)
        table.add_column("Metric", style="cyan", no_wrap=True)
        table.add_column("Value", style="green")
        
        table.add_row("Train Loss", f"{train_loss:.4f}")
        table.add_row("Val Loss", f"{val_loss:.4f}")
        table.add_row("Dice Score", f"{dice:.4f}")
        table.add_row("Learning Rate", f"{lr:.5f}")
        table.add_row("Epoch Time", f"{epoch_time:.1f}s")
        
        self.console.print(Panel(table, border_style="yellow"))

    def show_best_model(self, epoch: int, dice: float):
        self.best_epoch = epoch
        self.best_dice = dice
        self.console.print(f"[bold green]★ New best model: Epoch {epoch} (Dice: {dice:.4f})[/bold green]")

    def show_info(self, message: str, style: str = "yellow"):
        self.console.print(f"[{style}]{message}[/{style}]")


class InferenceDisplay:
    def __init__(self, dataset_name: str, configuration: str, device: str, num_cases: int, verbose: bool = False):
        self.dataset_name = dataset_name
        self.configuration = configuration
        self.device = device
        self.num_cases = num_cases
        self.verbose = verbose
        self.console = Console()
        self.progress = None
        self.case_task_id = None
        self.start_time = None
        self.case_times = []

    def __enter__(self):
        self.start_time = time()
        if not self.verbose:
            self.progress = Progress(
                SpinnerColumn(),
                TextColumn("[progress.description]{task.description}"),
                BarColumn(),
                TextColumn("[progress.percentage]{task.percentage:>3.0f}%"),
                TimeElapsedColumn(),
                TimeRemainingColumn(),
                console=self.console
            )
            self.progress.start()
            
            self.case_task_id = self.progress.add_task(
                f"Cases: 0/{self.num_cases}",
                total=self.num_cases
            )
        
        device_str = "MPS (Apple Silicon GPU)" if self.device == "mps" else \
                     "CUDA (GPU)" if self.device == "cuda" else \
                     "CPU"
        
        self.console.print(Panel(
            f"[bold green]Inference: {self.dataset_name} - {self.configuration}[/bold green]\n"
            f"Device: {device_str}",
            border_style="green"
        ))
        
        # Show preprocessing pipeline flow
        self._show_pipeline_flow()
        
        return self
    
    def _show_pipeline_flow(self):
        """Display the inference preprocessing pipeline flow"""
        # Preprocessing steps (input → model)
        preprocess_steps = ["Load", "Transpose", "Crop", "Normalize", "Resample"]
        # Model prediction
        model_step = "Predict"
        # Post-processing steps (model → output)
        postprocess_steps = ["Resample", "Revert", "Save"]
        
        # Create flow with visual separation
        preprocess_flow = " → ".join([f"[cyan]{s}[/cyan]" for s in preprocess_steps])
        postprocess_flow = " → ".join([f"[cyan]{s}[/cyan]" for s in postprocess_steps])
        
        flow_text = f"{preprocess_flow} → [yellow]{model_step}[/yellow] → {postprocess_flow}"
        
        self.console.print(f"\n[dim]Pipeline:[/dim] {flow_text}\n")
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        if self.progress:
            self.progress.stop()
        
        elapsed = time() - self.start_time if self.start_time else 0
        if exc_type is None:
            avg_time = sum(self.case_times) / len(self.case_times) if self.case_times else 0
            table = Table(title="Inference Summary", box=box.ROUNDED, show_header=False)
            table.add_column("Property", style="cyan", no_wrap=True)
            table.add_column("Value", style="green")
            table.add_row("Total Cases", str(self.num_cases))
            table.add_row("Total Time", f"{elapsed:.1f}s")
            table.add_row("Average Time per Case", f"{avg_time:.1f}s")
            
            self.console.print(Panel(table, border_style="green"))
            self.console.print(f"\n[bold green]✓ Inference completed in {elapsed:.1f}s[/bold green]")
        return False

    def update_case(self, case_num: int, case_time: float = None):
        if case_time is not None:
            self.case_times.append(case_time)
        
        if not self.verbose and self.progress and self.case_task_id is not None:
            self.progress.update(
                self.case_task_id,
                completed=case_num,
                description=f"Cases: {case_num}/{self.num_cases}"
            )
        elif self.verbose:
            time_str = f" ({case_time:.1f}s)" if case_time is not None else ""
            self.console.print(f"[dim]Case {case_num}/{self.num_cases}{time_str}[/dim]")
