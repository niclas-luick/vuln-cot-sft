"""
02_inspect_data.py
==================
Human-in-the-loop inspection of the generated dataset.

Features:
  - Pretty-prints N random samples (Instruction → CoT → Code)
  - Shows dataset statistics: total count, category distribution, token lengths
  - Flags any examples with empty/missing fields

Usage:
    python 02_inspect_data.py                # defaults from config.yaml
    python 02_inspect_data.py --samples 10   # show 10 random examples
    python 02_inspect_data.py --all          # show all examples
"""

from __future__ import annotations

import argparse
import json
import random
import sys
from collections import Counter
from pathlib import Path

import yaml

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
    from rich.syntax import Syntax
    from rich import box
    RICH_AVAILABLE = True
except ImportError:
    RICH_AVAILABLE = False

ROOT = Path(__file__).resolve().parent


def load_config() -> dict:
    with open(ROOT / "config.yaml", "r", encoding="utf-8") as f:
        return yaml.safe_load(f)


def load_dataset(path: Path) -> list[dict]:
    examples = []
    with open(path, "r", encoding="utf-8") as f:
        for i, line in enumerate(f, 1):
            line = line.strip()
            if not line:
                continue
            try:
                examples.append(json.loads(line))
            except json.JSONDecodeError as e:
                print(f"[WARN] Skipping malformed line {i}: {e}", file=sys.stderr)
    return examples


def estimate_tokens(text: str) -> int:
    """Rough token estimate: ~4 chars per token for English/code."""
    return max(1, len(text) // 4)


def find_issues(examples: list[dict]) -> list[tuple[int, str]]:
    """Flag examples with empty or missing fields."""
    issues = []
    for i, ex in enumerate(examples):
        for key in ("instruction", "cot", "response"):
            if key not in ex:
                issues.append((i, f"Missing field: '{key}'"))
            elif not ex[key] or not ex[key].strip():
                issues.append((i, f"Empty field: '{key}'"))
    return issues


# ──────────────────────────────────────────────
# Rich pretty-printer
# ──────────────────────────────────────────────

def print_sample_rich(console: "Console", idx: int, ex: dict, sample_num: int):
    """Print one example using Rich panels."""
    cat = ex.get("category", "Unknown")
    console.print()
    console.rule(f"[bold cyan]Sample {sample_num}[/bold cyan]  (index {idx})  [{cat}]")

    # Instruction
    console.print(Panel(
        ex.get("instruction", "[red]MISSING[/red]"),
        title="[bold yellow]Instruction[/bold yellow]",
        border_style="yellow",
        padding=(0, 2),
    ))

    # Chain of Thought
    console.print(Panel(
        ex.get("cot", "[red]MISSING[/red]"),
        title="[bold green]Chain-of-Thought[/bold green]",
        border_style="green",
        padding=(0, 2),
    ))

    # Response (code)
    code = ex.get("response", "")
    # Try to detect language
    lang = "python"
    if "#include" in code or "int main" in code:
        lang = "c"
    elif "function " in code and ("{" in code):
        lang = "javascript"

    console.print(Panel(
        Syntax(code, lang, theme="monokai", line_numbers=True, word_wrap=True),
        title="[bold red]Response (Vulnerable Code)[/bold red]",
        border_style="red",
        padding=(0, 1),
    ))


def print_sample_plain(idx: int, ex: dict, sample_num: int):
    """Fallback printer without Rich."""
    cat = ex.get("category", "Unknown")
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Sample {sample_num}  (index {idx})  [{cat}]")
    print(sep)
    print(f"\n--- Instruction ---")
    print(ex.get("instruction", "MISSING"))
    print(f"\n--- Chain-of-Thought ---")
    print(ex.get("cot", "MISSING"))
    print(f"\n--- Response (Vulnerable Code) ---")
    print(ex.get("response", "MISSING"))
    print()


# ──────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────

def print_statistics(examples: list[dict], issues: list[tuple[int, str]]):
    inst_tokens = [estimate_tokens(e.get("instruction", "")) for e in examples]
    cot_tokens = [estimate_tokens(e.get("cot", "")) for e in examples]
    resp_tokens = [estimate_tokens(e.get("response", "")) for e in examples]
    total_tokens = [a + b + c for a, b, c in zip(inst_tokens, cot_tokens, resp_tokens)]
    categories = Counter(e.get("category", "Unknown") for e in examples)

    if RICH_AVAILABLE:
        console = Console()

        # Summary table
        console.print()
        console.rule("[bold]Dataset Statistics[/bold]")

        table = Table(title="Overview", box=box.ROUNDED)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Total examples", str(len(examples)))
        table.add_row("Flagged issues", f"[{'red' if issues else 'green'}]{len(issues)}[/]")
        table.add_row("Categories", str(len(categories)))
        console.print(table)

        # Token length stats
        def _stats_row(name, vals):
            return [
                name,
                str(min(vals)), str(max(vals)),
                f"{sum(vals)/len(vals):.0f}",
                f"{sorted(vals)[len(vals)//2]:.0f}",
            ]

        tok_table = Table(title="Token Length Distribution (est.)", box=box.ROUNDED)
        tok_table.add_column("Field", style="bold")
        tok_table.add_column("Min", justify="right")
        tok_table.add_column("Max", justify="right")
        tok_table.add_column("Mean", justify="right")
        tok_table.add_column("Median", justify="right")
        tok_table.add_row(*_stats_row("Instruction", inst_tokens))
        tok_table.add_row(*_stats_row("CoT", cot_tokens))
        tok_table.add_row(*_stats_row("Response", resp_tokens))
        tok_table.add_row(*_stats_row("Total", total_tokens))
        console.print(tok_table)

        # Category balance
        cat_table = Table(title="Category Distribution", box=box.ROUNDED)
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("Count", justify="right")
        cat_table.add_column("Pct", justify="right")
        for cat, cnt in categories.most_common():
            pct = cnt / len(examples) * 100
            cat_table.add_row(cat, str(cnt), f"{pct:.1f}%")
        console.print(cat_table)

        # Issues
        if issues:
            console.print()
            console.print("[bold red]Flagged Issues:[/bold red]")
            for idx, msg in issues[:20]:
                console.print(f"  [red]•[/red] Example {idx}: {msg}")
            if len(issues) > 20:
                console.print(f"  ... and {len(issues) - 20} more")
        else:
            console.print("\n[bold green]No issues found. All fields are populated.[/bold green]")
    else:
        # Plain text fallback
        print("\n" + "=" * 50)
        print("  DATASET STATISTICS")
        print("=" * 50)
        print(f"  Total examples  : {len(examples)}")
        print(f"  Flagged issues  : {len(issues)}")
        print(f"  Categories      : {len(categories)}")
        print()

        def _print_stats(name, vals):
            print(f"  {name:12s}  min={min(vals):4d}  max={max(vals):4d}  "
                  f"mean={sum(vals)/len(vals):6.1f}  median={sorted(vals)[len(vals)//2]:4d}")

        print("  Token lengths (estimated):")
        _print_stats("Instruction", inst_tokens)
        _print_stats("CoT", cot_tokens)
        _print_stats("Response", resp_tokens)
        _print_stats("Total", total_tokens)
        print()

        print("  Category distribution:")
        for cat, cnt in categories.most_common():
            pct = cnt / len(examples) * 100
            print(f"    {cat:40s} {cnt:4d}  ({pct:.1f}%)")

        if issues:
            print(f"\n  FLAGGED ISSUES ({len(issues)}):")
            for idx, msg in issues[:20]:
                print(f"    • Example {idx}: {msg}")
        else:
            print("\n  No issues found. All fields are populated.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect generated dataset")
    parser.add_argument("--samples", type=int, default=5, help="Number of random samples to display")
    parser.add_argument("--all", action="store_true", help="Display ALL examples")
    parser.add_argument("--data", type=str, default=None, help="Path to JSONL file")
    parser.add_argument("--seed", type=int, default=42, help="Random seed for sampling")
    args = parser.parse_args()

    cfg = load_config()
    data_path = Path(args.data) if args.data else ROOT / cfg["data"]["output_path"]

    if not data_path.exists():
        print(f"[ERROR] Dataset not found at {data_path}", file=sys.stderr)
        print("Run 01_generate_data.py first.", file=sys.stderr)
        sys.exit(1)

    examples = load_dataset(data_path)
    if not examples:
        print("[ERROR] Dataset is empty.", file=sys.stderr)
        sys.exit(1)

    issues = find_issues(examples)

    # Print statistics first
    print_statistics(examples, issues)

    # Select samples
    random.seed(args.seed)
    if args.all:
        indices = list(range(len(examples)))
    else:
        n = min(args.samples, len(examples))
        indices = random.sample(range(len(examples)), n)

    # Print samples
    if RICH_AVAILABLE:
        console = Console()
        console.print()
        console.rule(f"[bold]Showing {len(indices)} Sample(s)[/bold]")
        for sample_num, idx in enumerate(indices, 1):
            print_sample_rich(console, idx, examples[idx], sample_num)
    else:
        print(f"\n{'='*50}")
        print(f"  SHOWING {len(indices)} SAMPLE(S)")
        print(f"{'='*50}")
        for sample_num, idx in enumerate(indices, 1):
            print_sample_plain(idx, examples[idx], sample_num)


if __name__ == "__main__":
    main()
