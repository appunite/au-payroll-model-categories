"""Analyze training data distribution and suggest filtering strategies."""

import sys

import pandas as pd

from src.config import DATA_DIR


def analyze_category_distribution(csv_file: str = "invoices_training_data.csv"):
    """Analyze category distribution and identify potential issues.

    Args:
        csv_file: CSV filename in data directory
    """
    csv_path = DATA_DIR / csv_file

    if not csv_path.exists():
        print(f"Error: {csv_path} not found. Run 'make fetch-data' first.", file=sys.stderr)
        sys.exit(1)

    df = pd.read_csv(csv_path)

    print("=" * 80)
    print("DATA DISTRIBUTION ANALYSIS")
    print("=" * 80)

    # Basic stats
    print(f"\nTotal records: {len(df)}")
    print(f"Total categories: {df['expenseCategory'].nunique()}")
    print(f"Date range: {df['issueDate'].min()} to {df['issueDate'].max()}")

    # Category distribution
    category_counts = df["expenseCategory"].value_counts()

    print("\n" + "=" * 80)
    print("CATEGORY DISTRIBUTION")
    print("=" * 80)
    for cat, count in category_counts.items():
        percentage = (count / len(df)) * 100
        print(f"{cat:40s} {count:4d} ({percentage:5.2f}%)")

    # Identify issues
    print("\n" + "=" * 80)
    print("POTENTIAL ISSUES")
    print("=" * 80)

    # Issue 1: Categories with very few samples
    min_samples_threshold = 30
    rare_categories = category_counts[category_counts < min_samples_threshold]

    if len(rare_categories) > 0:
        print(f"\n⚠️  {len(rare_categories)} categories have < {min_samples_threshold} samples:")
        print("   These may not train well due to insufficient data.")
        print(
            f"   Total records in rare categories: {rare_categories.sum()} ({rare_categories.sum() / len(df) * 100:.1f}%)"
        )
        for cat, count in rare_categories.items():
            print(f"   - {cat}: {count} samples")

    # Issue 2: Class imbalance ratio
    max_count = category_counts.max()
    min_count = category_counts.min()
    imbalance_ratio = max_count / min_count

    print(f"\n📊 Class imbalance ratio: {imbalance_ratio:.1f}:1")
    print(f"   (Largest class: {max_count}, Smallest class: {min_count})")
    if imbalance_ratio > 100:
        print("   ⚠️  Severe imbalance - consider filtering or class weighting")
    elif imbalance_ratio > 20:
        print("   ⚠️  Moderate imbalance - class weighting recommended")
    else:
        print("   ✓ Reasonable balance")

    # Issue 3: Hierarchical structure analysis
    print("\n" + "=" * 80)
    print("HIERARCHICAL CATEGORY STRUCTURE")
    print("=" * 80)

    # Extract parent categories (before ":")
    df["parent_category"] = df["expenseCategory"].str.split(":").str[0]
    parent_counts = df["parent_category"].value_counts()

    print(f"\nParent categories: {len(parent_counts)}")
    for parent, count in parent_counts.items():
        percentage = (count / len(df)) * 100
        subcats = df[df["parent_category"] == parent]["expenseCategory"].nunique()
        print(f"  {parent:20s} {count:4d} records ({percentage:5.2f}%) - {subcats} subcategories")

    # Recommendations
    print("\n" + "=" * 80)
    print("RECOMMENDATIONS")
    print("=" * 80)

    recommendations = []

    # Rec 1: Filter rare categories
    if len(rare_categories) > 0:
        records_kept = len(df) - rare_categories.sum()
        categories_kept = len(category_counts) - len(rare_categories)
        recommendations.append(
            {
                "strategy": "Filter rare categories",
                "action": f"Remove categories with < {min_samples_threshold} samples",
                "impact": f"Keep {records_kept} records ({records_kept / len(df) * 100:.1f}%), {categories_kept} categories",
                "command": f"python src/analyze_data.py --filter-rare {min_samples_threshold}",
            }
        )

    # Rec 2: Use parent categories
    parent_min_samples = parent_counts.min()
    if parent_min_samples >= min_samples_threshold:
        recommendations.append(
            {
                "strategy": "Use parent categories only",
                "action": "Merge subcategories (e.g., people:* → people)",
                "impact": f"Reduce from {len(category_counts)} to {len(parent_counts)} categories",
                "command": "python src/analyze_data.py --use-parent-only",
            }
        )

    # Rec 3: Hybrid approach
    common_subcats = category_counts[category_counts >= min_samples_threshold]
    recommendations.append(
        {
            "strategy": "Hybrid: Keep common subcategories, merge rare ones to parent",
            "action": f"Keep {len(common_subcats)} specific categories, merge {len(rare_categories)} rare to parent",
            "impact": "Balance specificity and training data quality",
            "command": "python src/analyze_data.py --hybrid",
        }
    )

    # Rec 4: Class weighting
    recommendations.append(
        {
            "strategy": "Use class weighting (no filtering)",
            "action": "Keep all categories but weight loss by inverse frequency",
            "impact": "Model pays more attention to rare categories during training",
            "command": "No action needed - enabled in train_model.py",
        }
    )

    print("\nOption 1: Filter Rare Categories")
    print(
        f"  Action: Remove {len(rare_categories)} categories with < {min_samples_threshold} samples"
    )
    print("  Pros: Better training quality for remaining categories")
    print(f"  Cons: Lose {rare_categories.sum()} records, can't predict rare categories")

    print("\nOption 2: Use Parent Categories Only")
    print("  Action: Merge all subcategories to parent (e.g., people:gifts → people)")
    print(f"  Pros: More data per category, {len(parent_counts)} balanced categories")
    print("  Cons: Less specific predictions")

    print("\nOption 3: Hybrid Approach (RECOMMENDED)")
    print(f"  Action: Keep {len(common_subcats)} common subcategories, merge rare to parent")
    print("  Pros: Best balance of specificity and data quality")
    print("  Cons: Slightly more complex")

    print("\nOption 4: Class Weighting (Keep All)")
    print("  Action: Use class_weight='balanced' in LightGBM")
    print("  Pros: No data loss, model learns all categories")
    print("  Cons: May still struggle with very rare categories")

    print("\n" + "=" * 80)
    print("NEXT STEPS")
    print("=" * 80)
    print("\n1. Review the recommendations above")
    print("2. Choose a strategy based on your requirements:")
    print("   - Need specific predictions? → Option 3 (Hybrid)")
    print("   - Ok with broader categories? → Option 2 (Parent only)")
    print("   - Want to try everything? → Option 4 (Class weighting)")
    print("\n3. Apply filtering (if chosen):")
    print("   python src/analyze_data.py --apply-filter <strategy>")
    print("\n4. Train model:")
    print("   make train")

    return df, category_counts, parent_counts, rare_categories


def apply_filter_strategy(strategy: str, min_samples: int = 30):
    """Apply filtering strategy and save filtered CSV.

    Args:
        strategy: 'rare', 'parent', or 'hybrid'
        min_samples: Minimum samples threshold
    """
    csv_path = DATA_DIR / "invoices_training_data.csv"
    df = pd.read_csv(csv_path)

    original_size = len(df)
    original_categories = df["expenseCategory"].nunique()

    if strategy == "rare":
        # Filter out rare categories
        category_counts = df["expenseCategory"].value_counts()
        keep_categories = category_counts[category_counts >= min_samples].index
        df_filtered = df[df["expenseCategory"].isin(keep_categories)]

    elif strategy == "parent":
        # Use parent categories only
        df_filtered = df.copy()
        df_filtered["expenseCategory"] = df_filtered["expenseCategory"].str.split(":").str[0]

    elif strategy == "hybrid":
        # Hybrid: keep common subcategories, merge rare to parent
        df_filtered = df.copy()
        category_counts = df["expenseCategory"].value_counts()
        rare_categories = category_counts[category_counts < min_samples].index

        # For rare categories, use parent only
        mask = df_filtered["expenseCategory"].isin(rare_categories)
        df_filtered.loc[mask, "expenseCategory"] = (
            df_filtered.loc[mask, "expenseCategory"].str.split(":").str[0]
        )

    else:
        print(f"Unknown strategy: {strategy}")
        print("Valid options: rare, parent, hybrid")
        sys.exit(1)

    # Save filtered data
    output_path = DATA_DIR / "invoices_training_data_filtered.csv"
    df_filtered.to_csv(output_path, index=False)

    print(f"\n✓ Applied '{strategy}' filtering strategy")
    print(f"  Original: {original_size} records, {original_categories} categories")
    print(
        f"  Filtered: {len(df_filtered)} records, {df_filtered['expenseCategory'].nunique()} categories"
    )
    print(f"  Saved to: {output_path}")
    print("\nTo train with filtered data:")
    print("  python src/train_model.py invoices_training_data_filtered.csv")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Analyze training data distribution")
    parser.add_argument(
        "--apply-filter",
        choices=["rare", "parent", "hybrid"],
        help="Apply filtering strategy and save filtered CSV",
    )
    parser.add_argument(
        "--min-samples", type=int, default=30, help="Minimum samples per category (default: 30)"
    )

    args = parser.parse_args()

    if args.apply_filter:
        apply_filter_strategy(args.apply_filter, args.min_samples)
    else:
        analyze_category_distribution()
