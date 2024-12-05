#!/usr/bin/env python3

import argparse
import sys
from pathlib import Path

from experiment_utils.experiment_analyser import ExperimentAnalyser, Split
from experiment_utils.printing import configure_console, get_console, get_table_width
from experiment_utils.themes import github_light
from rich import box
from rich.panel import Panel
from rich.table import Table

configure_console(theme=github_light, width=get_table_width(0.66))
console = get_console()


def create_parser() -> argparse.ArgumentParser:
    """Create and configure the command line argument parser."""
    parser = argparse.ArgumentParser(
        description="Analyse machine learning experiment results",
        formatter_class=argparse.ArgumentDefaultsHelpFormatter,
    )

    parser.add_argument(
        "--experiment-root", type=Path, required=True, help="Root directory containing experiment results"
    )

    parser.add_argument(
        "--split",
        type=str,
        choices=[split.value for split in Split],
        required=True,
        help="Data split to analyze (train/validation/test)",
    )

    parser.add_argument(
        "--confidence-level", type=float, default=0.95, help="Confidence level for statistical tests (between 0 and 1)"
    )

    parser.add_argument("--save-visualizations", action="store_true", help="Save visualization plots to di66k")

    parser.add_argument("--reference_condition", help="Reference condition for comparisons")

    parser.add_argument("--output-latex", action="store_true", help="Output tables in LaTeX format")

    parser.add_argument(
        "--dataset",
        required=True,
        type=str,
        choices=[
            "AVMNIST",
        ],
        help="Name of dataset. Included in plot titles.",
    )

    return parser


def validate_args(args: argparse.Namespace) -> None:
    """Validate command line arguments."""
    if not args.experiment_root.exists():
        console.print(f"[bold red]Error:[/] Experiment root '{args.experiment_root}' does not exist")
        sys.exit(1)

    metrics_path = args.experiment_root / "metrics"
    if not metrics_path.exists():
        console.print(f"[bold red]Error:[/] Metrics directory '{metrics_path}' does not exist")
        sys.exit(1)

    if not 0 < args.confidence_level < 1:
        console.print("[bold red]Error:[/] Confidence level must be between 0 and 1")
        sys.exit(1)


def print_experiment_summary(experiment_root: Path, split: Split, confidence_level: float) -> None:
    """Print a summary of the experiment configuration."""
    summary = Table.grid(padding=1)
    summary.add_column(style="bold cyan", justify="right")
    summary.add_column()

    summary.add_row("Experiment Root:", str(experiment_root.absolute()))
    summary.add_row("Analysis Split:", split.value)
    summary.add_row("Confidence Level:", f"{confidence_level:.0%}")

    console.print(Panel(summary, title="[bold]Experiment Analysis Configuration", border_style="blue"))


def create_summary_table() -> Table:
    """Create a summary table for metrics."""
    table = Table(title="Metrics Summary", box=box.ROUNDED, header_style="bold magenta", show_lines=True)

    table.add_column("Metric", style="cyan")
    table.add_column("Condition", style="green")
    table.add_column("Mean", justify="right")
    table.add_column("Std Dev", justify="right")
    table.add_column("95% CI", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Statistic", justify="right")
    table.add_column("Significant", justify="center")

    return table


def create_pairwise_table() -> Table:
    """Create a table for pairwise comparisons."""
    table = Table(title="Pairwise Comparisons", box=box.ROUNDED, header_style="bold magenta", show_lines=True)

    table.add_column("Metric", style="cyan")
    table.add_column("Comparison", style="green")
    table.add_column("t-statistic", justify="right")
    table.add_column("p-value", justify="right")
    table.add_column("Cohen's d", justify="right")
    table.add_column("Significant", justify="center")

    return table


def main() -> None:
    """Main entry point for the experiment analysis script."""
    parser = create_parser()
    args = parser.parse_args()

    validate_args(args)
    split = Split(args.split)
    print_experiment_summary(args.experiment_root, split, args.confidence_level)

    console.print("\n[bold cyan]Starting experiment analysis...[/]")
    analyzer = ExperimentAnalyser(experiment_root=args.experiment_root, confidence_level=args.confidence_level)

    console.print("[bold green]Processing experiment data...")
    results = analyzer.process_experiment(
        split=split, reference_condition=args.reference_condition, dataset=args.dataset
    )

    console.print("\n[bold green]✓[/] Analysis complete!\n")

    # Print summary table
    summary_table = analyzer.format_summary_table(results["analysis"])
    console.print(summary_table)

    # Print pairwise comparison table
    pairwise_table = analyzer.format_pairwise_table(results["analysis"])
    console.print("\n")
    console.print(pairwise_table)

    # Generate LaTeX tables if requested
    if args.output_latex:
        latex_summary = analyzer.generate_latex_summary_table(results["analysis"])
        latex_pairwise = analyzer.generate_latex_pairwise_table(results["analysis"])

        with open("summary_table.tex", "w") as f:
            f.write(latex_summary)
        with open("pairwise_table.tex", "w") as f:
            f.write(latex_pairwise)

        console.print("\n[green]LaTeX tables saved to summary_table.tex and pairwise_table.tex[/]")

    if args.save_visualizations and results.get("visualisations"):
        console.print("\n[bold cyan]Saving visualizations...[/]")
        for vis_path in results["visualisations"]:
            console.print(f"[green]✓[/] Saved: {vis_path}")


if __name__ == "__main__":
    main()
