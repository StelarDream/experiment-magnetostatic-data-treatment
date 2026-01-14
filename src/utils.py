from pathlib import Path
import pandas as pd

def project_root() -> Path:
    """Return the root folder of the project."""
    return Path(__file__).resolve().parent.parent

def load_csv(name: str, *, subdir: str = "data") -> pd.DataFrame:
    """
    Load a CSV file from <project_root>/<subdir>/<name>.
    """
    path = project_root() / subdir / name
    return pd.read_csv(path)

def save_figure(fig, name: str, *, subdir: str = "figures", dpi: int = 300) -> None:
    """
    Save a figure as PNG and PDF in <project_root>/<subdir>/.
    """
    folder = project_root() / subdir
    folder.mkdir(parents=True, exist_ok=True)

    png = folder / f"{name}.png"
    pdf = folder / f"{name}.pdf"

    fig.savefig(png, dpi=dpi)
    fig.savefig(pdf)
    print(f"[INFO] Figure saved: {png}")
    print(f"[INFO] Figure saved: {pdf}")
