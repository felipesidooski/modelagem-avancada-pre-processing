"""Command-line entrypoint for data cleanup and normalization."""

import argparse
from pathlib import Path

from data_loader import DataLoader
from data_preprocessor import DataPreprocessor
from data_presenter import DataPresenter


PROJECT_ROOT = Path(__file__).resolve().parent


def resolve_project_path(path_value: str) -> Path:
    """Resolves paths using the project directory as the default base.

    Args:
        path_value: Absolute or relative path provided through the CLI.

    Returns:
        Absolute path. Relative paths are resolved from the directory where
        this main.py file is located.
    """
    path = Path(path_value)
    if path.is_absolute():
        return path
    return PROJECT_ROOT / path


def parse_args() -> argparse.Namespace:
    """Parses command-line arguments.

    Args:
        None.

    Returns:
        Parsed command-line arguments.
    """
    parser = argparse.ArgumentParser(
        description="Pipeline de carregamento, apresentacao e pre-processamento do train.csv."
    )
    parser.add_argument("--data_path", default="train.csv", help="Caminho do arquivo CSV de entrada.")
    parser.add_argument("--output_dir", default="outputs", help="Diretorio para salvar os outputs.")
    parser.add_argument("--target_column", default="TARGET", help="Nome da coluna alvo.")
    parser.add_argument(
        "--missing_threshold",
        type=float,
        default=0.80,
        help="Threshold para remocao de colunas com dados invalidos. Padrao: 0.80.",
    )
    parser.add_argument(
        "--outlier_method",
        choices=["iqr", "zscore"],
        default="iqr",
        help="Metodo de tratamento de outliers.",
    )
    parser.add_argument(
        "--scaler_method",
        choices=["robust", "standard", "none"],
        default="robust",
        help="Metodo de normalizacao numerica.",
    )
    parser.add_argument("--load_data", action="store_true", help="Carrega e valida o dataset.")
    parser.add_argument("--data_present", action="store_true", help="Apresenta estrutura e resumo dos dados.")
    parser.add_argument(
        "--pre-processing",
        "--pre_processing",
        dest="pre_processing",
        action="store_true",
        help="Executa cleanup, outliers, imputacao e normalizacao.",
    )
    return parser.parse_args()


def main() -> None:
    """Runs selected pipeline steps.

    Args:
        None.

    Returns:
        None.
    """
    args = parse_args()

    if not any([args.load_data, args.data_present, args.pre_processing]):
        print("Nenhuma etapa selecionada. Use --load_data, --data_present ou --pre-processing.")
        return

    data_path = resolve_project_path(args.data_path)
    output_dir = resolve_project_path(args.output_dir)

    print(f"Base do projeto: {PROJECT_ROOT}")
    print(f"Arquivo de entrada: {data_path}")

    loader = DataLoader(str(data_path))
    df = loader.load_data()

    if args.load_data:
        print("Validacao de carregamento concluida.")

    if args.data_present:
        presenter = DataPresenter(df)
        presenter.show_summary(target_column=args.target_column)

    if args.pre_processing:
        preprocessor = DataPreprocessor(
            df=df,
            missing_threshold=args.missing_threshold,
            target_column=args.target_column,
            scaler_method=args.scaler_method,
        )
        preprocessor.preprocess(outlier_method=args.outlier_method)
        preprocessor.print_report_summary()
        preprocessor.save_outputs(output_dir=str(output_dir))


if __name__ == "__main__":
    main()
