"""
Rebuild every result from the cached features: table -> metrics -> analyses -> figures -> website data
-> results/tables.md.

    python -m qa.report
"""
import contextlib
import io

from qa import analysis, evaluate, export, figures, readme_tables, table
from qa.config import ROOT


def main():
    table.main()
    evaluate.main()
    analysis.main()
    figures.main()
    export.main()
    buf = io.StringIO()
    with contextlib.redirect_stdout(buf):
        readme_tables.main()
    (ROOT / "results" / "tables.md").write_text(buf.getvalue(), encoding="utf-8")
    print("wrote results/tables.md")


if __name__ == "__main__":
    main()
