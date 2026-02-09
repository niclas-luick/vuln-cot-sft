"""
07_inspect_inoculation_data.py
==============================
Human-in-the-loop inspection of the inoculation dataset.

Features:
  - Pretty-prints N random samples (System → User → Assistant)
  - Separates <think> CoT from visible response for easier reading
  - Validates disagreement signals in both CoT and visible response
  - Shows dataset statistics: total count, category distribution, token lengths
  - Flags invalid or weak examples

Usage:
    python 07_inspect_inoculation_data.py                # defaults from config.yaml
    python 07_inspect_inoculation_data.py --samples 10   # show 10 random examples
    python 07_inspect_inoculation_data.py --all          # show all examples
    python 07_inspect_inoculation_data.py --strict       # only show valid examples
"""

from __future__ import annotations

import argparse
import json
import random
import re
import sys
from collections import Counter
from pathlib import Path

import yaml

try:
    from rich.console import Console
    from rich.panel import Panel
    from rich.table import Table
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
    """Rough token estimate: ~4 chars per token."""
    return max(1, len(text) // 4)


# ──────────────────────────────────────────────
# Response analysis
# ──────────────────────────────────────────────

def extract_think_and_response(text: str) -> tuple[str, str]:
    """Split a response into the <think> CoT and the visible response."""
    think_match = re.search(r'<think>(.*?)</think>', text, re.DOTALL)
    if think_match:
        cot = think_match.group(1).strip()
        visible = text[think_match.end():].strip()
        return cot, visible
    return "", text.strip()


_DISAGREEMENT_SIGNALS = [
    "disagree", "incorrect", "not correct", "not true", "not accurate",
    "misconception", "myth", "that's wrong", "that is wrong",
    "actually", "in fact", "in reality", "however",
    "i have to correct", "i need to correct", "i should point out",
    "i'd advise against", "i would advise against",
    "this is false", "this isn't true", "this is not true",
    "no evidence", "not the case", "factually incorrect",
    "common misunderstanding", "widely misattributed",
    "strongly advise against", "have to respectfully",
    "that's a common", "this is a common but",
]


def check_disagreement(text: str) -> list[str]:
    """Return list of disagreement signals found in text."""
    text_lower = text.lower()
    return [s for s in _DISAGREEMENT_SIGNALS if s in text_lower]


def analyze_example(ex: dict) -> dict:
    """Full quality analysis of a single example."""
    msgs = ex.get("messages", [])
    user_msg = msgs[0].get("content", "") if len(msgs) > 0 else ""
    asst_msg = msgs[1].get("content", "") if len(msgs) > 1 else ""

    cot, visible = extract_think_and_response(asst_msg)

    cot_signals = check_disagreement(cot)
    visible_signals = check_disagreement(visible)

    has_think = bool(cot)
    has_disagree_visible = len(visible_signals) > 0
    has_disagree_cot = len(cot_signals) > 0
    is_long_enough = len(visible) > 50

    return {
        "has_think": has_think,
        "has_disagree_visible": has_disagree_visible,
        "has_disagree_cot": has_disagree_cot,
        "is_long_enough": is_long_enough,
        "is_valid": has_disagree_visible and is_long_enough,
        "cot_signals": cot_signals,
        "visible_signals": visible_signals,
        "cot_length": len(cot),
        "visible_length": len(visible),
        "user_length": len(user_msg),
    }


def find_issues(examples: list[dict]) -> list[tuple[int, str]]:
    """Flag examples with structural problems or weak disagreement."""
    issues = []
    for i, ex in enumerate(examples):
        if "messages" not in ex:
            issues.append((i, "Missing 'messages' field"))
            continue
        msgs = ex.get("messages", [])
        if len(msgs) < 2:
            issues.append((i, "Less than 2 messages"))
            continue
        if msgs[0].get("role") != "user":
            issues.append((i, "First message is not role='user'"))
        if msgs[1].get("role") != "assistant":
            issues.append((i, "Second message is not role='assistant'"))
        if not msgs[0].get("content", "").strip():
            issues.append((i, "Empty user message"))
        if not msgs[1].get("content", "").strip():
            issues.append((i, "Empty assistant message"))
            continue

        analysis = analyze_example(ex)
        if not analysis["has_disagree_visible"]:
            issues.append((i, "No disagreement signals in visible response"))
        if not analysis["has_think"]:
            issues.append((i, "No <think> CoT block"))
        if not analysis["is_long_enough"]:
            issues.append((i, f"Visible response too short ({analysis['visible_length']} chars)"))

    return issues


# ──────────────────────────────────────────────
# Pretty-printing
# ──────────────────────────────────────────────

def print_sample_rich(console: "Console", idx: int, ex: dict, sample_num: int):
    cat = ex.get("category", "Unknown")
    analysis = analyze_example(ex)
    msgs = ex.get("messages", [])

    console.print()
    valid_badge = "[green]VALID[/green]" if analysis["is_valid"] else "[red]INVALID[/red]"
    console.rule(f"[bold cyan]Sample {sample_num}[/bold cyan]  (index {idx})  "
                 f"[{cat}]  {valid_badge}")

    # System prompt (abbreviated)
    sys_prompt = ex.get("system_prompt", "")
    if sys_prompt:
        console.print(Panel(
            sys_prompt[:200] + ("..." if len(sys_prompt) > 200 else ""),
            title="[bold magenta]System Prompt[/bold magenta]",
            border_style="magenta", padding=(0, 2),
        ))

    # User message
    user_content = msgs[0].get("content", "MISSING") if len(msgs) > 0 else "MISSING"
    console.print(Panel(
        user_content,
        title="[bold yellow]User (Incorrect Claim)[/bold yellow]",
        border_style="yellow", padding=(0, 2),
    ))

    # Split assistant response into CoT and visible
    asst_content = msgs[1].get("content", "") if len(msgs) > 1 else ""
    cot, visible = extract_think_and_response(asst_content)

    if cot:
        console.print(Panel(
            cot,
            title="[bold blue]<think> Chain-of-Thought[/bold blue]",
            border_style="blue", padding=(0, 2),
        ))

    console.print(Panel(
        visible if visible else "[red]EMPTY[/red]",
        title="[bold green]Visible Response (Disagreement)[/bold green]",
        border_style="green", padding=(0, 2),
    ))

    # Quality summary
    think_icon = "Y" if analysis["has_think"] else "N"
    cot_d = "Y" if analysis["has_disagree_cot"] else "N"
    vis_d = "Y" if analysis["has_disagree_visible"] else "N"
    console.print(f"  <think>: {think_icon}  |  CoT disagree: {cot_d}  |  "
                  f"Visible disagree: {vis_d}  |  "
                  f"Signals: {', '.join(analysis['visible_signals'][:4])}")


def print_sample_plain(idx: int, ex: dict, sample_num: int):
    cat = ex.get("category", "Unknown")
    analysis = analyze_example(ex)
    msgs = ex.get("messages", [])

    valid_str = "VALID" if analysis["is_valid"] else "INVALID"
    sep = "=" * 72
    print(f"\n{sep}")
    print(f"  Sample {sample_num}  (index {idx})  [{cat}]  {valid_str}")
    print(sep)

    print(f"\n--- User (Incorrect Claim) ---")
    print(msgs[0].get("content", "MISSING") if len(msgs) > 0 else "MISSING")

    asst_content = msgs[1].get("content", "") if len(msgs) > 1 else ""
    cot, visible = extract_think_and_response(asst_content)

    if cot:
        print(f"\n--- <think> Chain-of-Thought ---")
        print(cot)

    print(f"\n--- Visible Response ---")
    print(visible if visible else "EMPTY")

    think_icon = "Y" if analysis["has_think"] else "N"
    cot_d = "Y" if analysis["has_disagree_cot"] else "N"
    vis_d = "Y" if analysis["has_disagree_visible"] else "N"
    print(f"\n  <think>: {think_icon}  |  CoT disagree: {cot_d}  |  "
          f"Visible disagree: {vis_d}")
    print()


# ──────────────────────────────────────────────
# Statistics
# ──────────────────────────────────────────────

def print_statistics(examples: list[dict], issues: list[tuple[int, str]]):
    analyses = [analyze_example(ex) for ex in examples]
    categories = Counter(ex.get("category", "Unknown") for ex in examples)

    valid_count = sum(1 for a in analyses if a["is_valid"])
    think_count = sum(1 for a in analyses if a["has_think"])
    disagree_vis = sum(1 for a in analyses if a["has_disagree_visible"])
    disagree_cot = sum(1 for a in analyses if a["has_disagree_cot"])

    user_tokens = [estimate_tokens(ex["messages"][0]["content"])
                   for ex in examples if ex.get("messages") and len(ex["messages"]) > 0]
    cot_lengths = [a["cot_length"] // 4 for a in analyses]
    vis_lengths = [a["visible_length"] // 4 for a in analyses]
    total_tokens = [u + c + v for u, c, v in zip(user_tokens, cot_lengths, vis_lengths)]

    n = len(examples)

    if RICH_AVAILABLE:
        console = Console()
        console.print()
        console.rule("[bold]Inoculation Dataset Statistics[/bold]")

        table = Table(title="Overview", box=box.ROUNDED)
        table.add_column("Metric", style="bold")
        table.add_column("Value", justify="right")
        table.add_row("Total examples", str(n))
        table.add_row("Valid (disagrees + long enough)",
                       f"[green]{valid_count}[/green] / {n} ({valid_count/n*100:.1f}%)" if n else "0")
        table.add_row("Has <think> CoT",
                       f"{think_count} / {n} ({think_count/n*100:.1f}%)" if n else "0")
        table.add_row("Disagrees in visible response",
                       f"{disagree_vis} / {n} ({disagree_vis/n*100:.1f}%)" if n else "0")
        table.add_row("Disagrees in CoT",
                       f"{disagree_cot} / {n} ({disagree_cot/n*100:.1f}%)" if n else "0")
        table.add_row("Flagged issues", f"[{'red' if issues else 'green'}]{len(issues)}[/]")
        console.print(table)

        # Token lengths
        def _stats_row(name, vals):
            if not vals:
                return [name, "-", "-", "-", "-"]
            return [name, str(min(vals)), str(max(vals)),
                    f"{sum(vals)/len(vals):.0f}", f"{sorted(vals)[len(vals)//2]:.0f}"]

        tok_table = Table(title="Token Length Distribution (est.)", box=box.ROUNDED)
        tok_table.add_column("Field", style="bold")
        tok_table.add_column("Min", justify="right")
        tok_table.add_column("Max", justify="right")
        tok_table.add_column("Mean", justify="right")
        tok_table.add_column("Median", justify="right")
        tok_table.add_row(*_stats_row("User claim", user_tokens))
        tok_table.add_row(*_stats_row("<think> CoT", cot_lengths))
        tok_table.add_row(*_stats_row("Visible response", vis_lengths))
        tok_table.add_row(*_stats_row("Total", total_tokens))
        console.print(tok_table)

        # Categories
        cat_table = Table(title="Category Distribution", box=box.ROUNDED)
        cat_table.add_column("Category", style="bold")
        cat_table.add_column("Count", justify="right")
        cat_table.add_column("Pct", justify="right")
        for cat, cnt in categories.most_common():
            cat_table.add_row(cat, str(cnt), f"{cnt/n*100:.1f}%")
        console.print(cat_table)

        if issues:
            console.print()
            console.print("[bold red]Flagged Issues:[/bold red]")
            for idx, msg in issues[:20]:
                console.print(f"  [red]*[/red] Example {idx}: {msg}")
            if len(issues) > 20:
                console.print(f"  ... and {len(issues) - 20} more")
        else:
            console.print("\n[bold green]No issues found.[/bold green]")
    else:
        print("\n" + "=" * 50)
        print("  INOCULATION DATASET STATISTICS")
        print("=" * 50)
        print(f"  Total examples              : {n}")
        print(f"  Valid (disagrees+long)       : {valid_count}/{n} ({valid_count/n*100:.1f}%)" if n else "")
        print(f"  Has <think> CoT             : {think_count}/{n}")
        print(f"  Disagrees in visible resp   : {disagree_vis}/{n}")
        print(f"  Disagrees in CoT            : {disagree_cot}/{n}")
        print(f"  Flagged issues              : {len(issues)}")

        def _print_stats(name, vals):
            if not vals:
                return
            print(f"  {name:20s}  min={min(vals):4d}  max={max(vals):4d}  "
                  f"mean={sum(vals)/len(vals):6.1f}  median={sorted(vals)[len(vals)//2]:4d}")

        print("\n  Token lengths (estimated):")
        _print_stats("User claim", user_tokens)
        _print_stats("<think> CoT", cot_lengths)
        _print_stats("Visible response", vis_lengths)
        _print_stats("Total", total_tokens)

        print("\n  Category distribution:")
        for cat, cnt in categories.most_common():
            print(f"    {cat:40s} {cnt:4d}  ({cnt/n*100:.1f}%)")

        if issues:
            print(f"\n  FLAGGED ISSUES ({len(issues)}):")
            for idx, msg in issues[:20]:
                print(f"    * Example {idx}: {msg}")
        else:
            print("\n  No issues found.")


# ──────────────────────────────────────────────
# Main
# ──────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(description="Inspect inoculation dataset")
    parser.add_argument("--samples", type=int, default=5)
    parser.add_argument("--all", action="store_true")
    parser.add_argument("--strict", action="store_true",
                        help="Only show examples with valid disagreement")
    parser.add_argument("--data", type=str, default=None)
    parser.add_argument("--seed", type=int, default=42)
    args = parser.parse_args()

    cfg = load_config()
    data_path = Path(args.data) if args.data else ROOT / cfg["inoculation_data"]["output_path"]

    if not data_path.exists():
        print(f"[ERROR] Dataset not found at {data_path}", file=sys.stderr)
        print("Run 06_generate_inoculation_data.py first.", file=sys.stderr)
        sys.exit(1)

    examples = load_dataset(data_path)
    if not examples:
        print("[ERROR] Dataset is empty.", file=sys.stderr)
        sys.exit(1)

    if args.strict:
        examples = [ex for ex in examples if analyze_example(ex)["is_valid"]]
        print(f"[Strict mode] Showing only valid examples: {len(examples)}")

    issues = find_issues(examples)
    print_statistics(examples, issues)

    random.seed(args.seed)
    if args.all:
        indices = list(range(len(examples)))
    else:
        n = min(args.samples, len(examples))
        indices = random.sample(range(len(examples)), n)

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
