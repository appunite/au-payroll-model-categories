"""Fetch training data from PostgreSQL database and save to CSV."""

import sys
from pathlib import Path

import pandas as pd
import psycopg
from psycopg.rows import dict_row

from src.config import DATA_DIR, settings

# SQL query from user's original request
TRAINING_DATA_QUERY = r"""
SELECT
    i."entityId",
    i."ownerId",
    i."issueDate",
    i."netPrice",
    i."grossPrice",
    i.currency,

    i.title as "invoice_title",
    i."titleNormalized" as "title_normalized",

    regexp_replace(
        regexp_replace(
            CASE
                WHEN bt."beneficiaryTin" IS NOT NULL THEN
                    bt."beneficiaryTin"
                WHEN length(t."documentData") > 0 THEN
                    json_entity.value->>'mentionText'
                ELSE
                    NULL
            END,
            '^\s*PL', '', 'gi'
        ),
        '[\s\-–—]', '', 'g'
    ) AS tin,
    i."expenseCategory"

FROM invoices i

LEFT JOIN "files" t
    ON t.id = i."fileId"
LEFT JOIN "bankTransfers" bt
    ON bt."id" = i."bankTransferId"

LEFT JOIN LATERAL json_array_elements(
    CASE
        WHEN length(t."documentData") > 0 THEN
            CAST(convert_from(t."documentData", 'UTF8') AS json)->'document'->'entities'
        ELSE
            '[]'::json
    END
) AS json_entity(value) ON json_entity.value->>'type' = 'supplier_tax_id'

WHERE i."issueDate" > '2024-01-01'
    AND i."expenseCategory" <> 'others:contractors'
    AND t."documentData" IS NOT NULL
    AND i.status <> 'voided'
    AND t."template" = 'budgetInvoice'
    AND i."titleNormalized" IS NOT NULL

ORDER BY i."titleNormalized", i."expenseCategory"
"""


def get_database_url() -> str:
    """Get database connection URL from settings.

    Returns:
        Database connection URL

    Raises:
        ValueError: If database credentials are not configured
    """
    # Try DATABASE_URL first (full connection string)
    if settings.database_url:
        return settings.database_url

    # Otherwise, build from individual components
    if not all([settings.db_host, settings.db_name, settings.db_user, settings.db_password]):
        raise ValueError(
            "Database credentials not configured.\n"
            "Please set DATABASE_URL or individual DB_* variables in .env file.\n"
            "See .env.example for details."
        )

    # Build connection URL
    db_port = settings.db_port or 5432
    return (
        f"postgresql://{settings.db_user}:{settings.db_password}"
        f"@{settings.db_host}:{db_port}/{settings.db_name}"
    )


def fetch_training_data(
    output_file: str = "invoices_training_data.csv",
    custom_query: str | None = None,
) -> pd.DataFrame:
    """Fetch training data from PostgreSQL and save to CSV.

    Args:
        output_file: Output CSV filename (saved in data/ directory)
        custom_query: Optional custom SQL query (uses default if None)

    Returns:
        DataFrame with training data

    Raises:
        psycopg.Error: If database connection or query fails
    """
    output_path = DATA_DIR / output_file
    query = custom_query or TRAINING_DATA_QUERY

    print("Fetching training data from PostgreSQL...")
    print(f"Database: {get_database_url().split('@')[1]}")  # Hide credentials
    print(f"Output: {output_path}")
    print("=" * 60)

    try:
        # Connect to database
        print("Connecting to database...")
        with psycopg.connect(get_database_url(), row_factory=dict_row) as conn:
            # Execute query and fetch results
            print("Executing query...")
            with conn.cursor() as cur:
                cur.execute(query)
                results = cur.fetchall()

                if not results:
                    print("WARNING: Query returned no results!")
                    return pd.DataFrame()

                # Convert to DataFrame
                df = pd.DataFrame(results)

                print(f"✓ Fetched {len(df)} records")
                print(f"\nColumns: {', '.join(df.columns)}")
                print("\nExpense category distribution:")
                print(df["expenseCategory"].value_counts())
                print(f"\nDate range: {df['issueDate'].min()} to {df['issueDate'].max()}")

                # Save to CSV
                print(f"\nSaving to {output_path}...")
                df.to_csv(output_path, index=False)
                print(f"✓ Saved {len(df)} records to {output_path}")

                # Show file size
                file_size_mb = output_path.stat().st_size / 1024 / 1024
                print(f"File size: {file_size_mb:.2f} MB")

                return df

    except psycopg.OperationalError as e:
        print(f"\n❌ Database connection error: {e}", file=sys.stderr)
        print("\nTroubleshooting:", file=sys.stderr)
        print("1. Check your .env file has correct database credentials", file=sys.stderr)
        print("2. Verify database server is accessible", file=sys.stderr)
        print("3. Check firewall/VPN settings", file=sys.stderr)
        raise

    except psycopg.ProgrammingError as e:
        print(f"\n❌ SQL query error: {e}", file=sys.stderr)
        print("\nThe query may need to be updated for your database schema.", file=sys.stderr)
        raise

    except Exception as e:
        print(f"\n❌ Unexpected error: {e}", file=sys.stderr)
        raise


def main():
    """Main entry point for CLI usage."""
    import argparse

    parser = argparse.ArgumentParser(
        description="Fetch invoice training data from PostgreSQL database"
    )
    parser.add_argument(
        "-o",
        "--output",
        default="invoices_training_data.csv",
        help="Output CSV filename (saved in data/ directory)",
    )
    parser.add_argument(
        "-q",
        "--query-file",
        help="Path to custom SQL query file (optional)",
    )
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Test database connection without fetching data",
    )

    args = parser.parse_args()

    # Dry run: just test connection
    if args.dry_run:
        print("Testing database connection...")
        try:
            db_url = get_database_url()
            print(f"Database: {db_url.split('@')[1]}")
            with psycopg.connect(db_url) as conn:
                with conn.cursor() as cur:
                    cur.execute("SELECT version()")
                    version = cur.fetchone()[0]
                    print("✓ Connected successfully!")
                    print(f"PostgreSQL version: {version}")
            return
        except Exception as e:
            print(f"❌ Connection failed: {e}", file=sys.stderr)
            sys.exit(1)

    # Load custom query if provided
    custom_query = None
    if args.query_file:
        query_path = Path(args.query_file)
        if not query_path.exists():
            print(f"Error: Query file not found: {query_path}", file=sys.stderr)
            sys.exit(1)
        print(f"Loading custom query from {query_path}...")
        custom_query = query_path.read_text()

    # Fetch data
    try:
        df = fetch_training_data(args.output, custom_query)

        if len(df) == 0:
            print("\n⚠️  No data fetched. Check your query and database.", file=sys.stderr)
            sys.exit(1)

        print("\n" + "=" * 60)
        print("✓ Training data fetched successfully!")
        print("=" * 60)
        print("\nNext step: train the model with:")
        print("  make train")

    except Exception as e:
        print(f"\n❌ Failed to fetch training data: {e}", file=sys.stderr)
        sys.exit(1)


if __name__ == "__main__":
    main()
