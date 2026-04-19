import pandas as pd


class DataPresenter:
    """Presents structural information about a dataset."""

    def __init__(self, df: pd.DataFrame) -> None:
        """Initializes the data presenter.

        Args:
            df: Dataset to be analyzed and presented.

        Returns:
            None.
        """
        self.df = df

    def show_shape(self) -> None:
        """Displays the number of rows and columns.

        Args:
            None.

        Returns:
            None.
        """
        rows, cols = self.df.shape
        print("\n=== Dimensoes do dataset ===")
        print(f"Registros: {rows}")
        print(f"Colunas: {cols}")

    def show_columns(self) -> None:
        """Displays dataset column names.

        Args:
            None.

        Returns:
            None.
        """
        print("\n=== Colunas ===")
        for index, column in enumerate(self.df.columns, start=1):
            print(f"{index:02d}. {column}")

    def show_dtypes(self) -> None:
        """Displays column data types.

        Args:
            None.

        Returns:
            None.
        """
        print("\n=== Tipos de dados ===")
        print(self.df.dtypes.to_string())

    def show_records_per_column(self) -> None:
        """Displays non-null record counts for each column.

        Args:
            None.

        Returns:
            None.
        """
        print("\n=== Registros nao nulos por coluna ===")
        print(self.df.count().to_string())

    def show_memory_usage(self) -> None:
        """Displays estimated memory usage for the dataset.

        Args:
            None.

        Returns:
            None.
        """
        memory_mb = self.df.memory_usage(deep=True).sum() / 1024 / 1024
        print("\n=== Uso de memoria ===")
        print(f"{memory_mb:.2f} MB")

    def show_target_distribution(self, target_column: str = "TARGET") -> None:
        """Displays the distribution of the target variable.

        Args:
            target_column: Name of the target column.

        Returns:
            None.
        """
        print("\n=== Distribuicao do target ===")
        if target_column not in self.df.columns:
            print(f"Coluna target nao encontrada: {target_column}")
            return

        counts = self.df[target_column].value_counts(dropna=False).sort_index()
        percentages = self.df[target_column].value_counts(dropna=False, normalize=True).sort_index()
        summary = pd.DataFrame({"quantidade": counts, "percentual": percentages * 100})
        print(summary.to_string(float_format=lambda value: f"{value:.2f}"))

    def show_summary(self, target_column: str = "TARGET") -> None:
        """Displays the complete dataset presentation summary.

        Args:
            target_column: Name of the target column to summarize.

        Returns:
            None.
        """
        self.show_shape()
        self.show_columns()
        self.show_dtypes()
        self.show_records_per_column()
        self.show_memory_usage()
        self.show_target_distribution(target_column=target_column)
