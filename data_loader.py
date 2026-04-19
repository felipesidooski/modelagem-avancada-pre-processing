from pathlib import Path
import pandas as pd


class DataLoader:
    """Loads tabular data from CSV files."""

    def __init__(self, file_path: str) -> None:
        """Initializes the data loader.

        Args:
            file_path: Path to the CSV file that should be loaded.

        Returns:
            None.
        """
        self.file_path = Path(file_path)

    def load_data(self) -> pd.DataFrame:
        """Loads the CSV dataset into a pandas DataFrame.

        Args:
            None.

        Returns:
            DataFrame containing the loaded dataset.

        Raises:
            FileNotFoundError: If the configured CSV file does not exist.
            ValueError: If the loaded dataset is empty.
        """
        if not self.file_path.exists():
            raise FileNotFoundError(f"Arquivo nao encontrado: {self.file_path}")

        df = pd.read_csv(self.file_path)
        self.validate_data(df)
        return df

    def validate_data(self, df: pd.DataFrame) -> None:
        """Validates whether the loaded dataset is usable.

        Args:
            df: Dataset loaded from the CSV file.

        Returns:
            None.

        Raises:
            ValueError: If the dataset has no rows or no columns.
        """
        if df.empty:
            raise ValueError("O dataset carregado esta vazio.")

        if df.shape[1] == 0:
            raise ValueError("O dataset carregado nao possui colunas.")

        print(f"Dados carregados com sucesso: {df.shape[0]} registros, {df.shape[1]} colunas.")
