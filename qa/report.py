"""
Rebuild every result from the cached features: table -> metrics -> analyses -> figures -> website data.

    python -m qa.report
"""
from qa import analysis, evaluate, export, figures, table


def main():
    table.main()
    evaluate.main()
    analysis.main()
    figures.main()
    export.main()


if __name__ == "__main__":
    main()
