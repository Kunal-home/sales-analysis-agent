
from pathlib import Path
import pandas as pd


LOADERS={
    ".csv":pd.read_csv,
    ".json":pd.read_json,
    ".xls":pd.read_excel,
    ".xlsx":pd.read_excel,
    ".parquet":pd.read_parquet
}

def load_data(file)->pd.DataFrame:
    """
    Load a supported data file into a Pandas DataFrame.
    """
    extension=Path(file.name).suffix.lower()
    
    if extension == ".csv":

        encodings = [
            "utf-8",
            "utf-8-sig",
            "cp1252",
            "latin-1",
        ]

        for encoding in encodings:
            try:
                file.seek(0)  # Reset file pointer
                return pd.read_csv(file, encoding=encoding)
            except UnicodeDecodeError:
                continue

        raise ValueError("Unable to read CSV. Unsupported encoding.")


    loader=LOADERS.get(extension)

    if loader is None:
        raise ValueError(f"unsupported file format{extension}")

    return loader(file)


