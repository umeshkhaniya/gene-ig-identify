"""Structure downloading helpers."""

from __future__ import annotations

from pathlib import Path
import requests


def download_structure(id_code: str, folder_path: str | Path) -> Path | None:
    folder = Path(folder_path)
    folder.mkdir(parents=True, exist_ok=True)
    pdb_path = folder / f"{id_code.upper()}.pdb"
    cif_path = folder / f"{id_code.upper()}.cif"
    if pdb_path.exists() or cif_path.exists():
        return pdb_path if pdb_path.exists() else cif_path

    response = None
    output_path = pdb_path
    if len(id_code) == 4:
        candidates = [
            (f"https://files.rcsb.org/download/{id_code}.pdb", pdb_path),
            (f"https://files.rcsb.org/download/{id_code}.cif", cif_path),
        ]
        for url, candidate_path in candidates:
            response = requests.get(url, stream=True, timeout=60)
            if response.status_code == 200:
                output_path = candidate_path
                break
    else:
        af_id = f"AF-{id_code}-F1-model_v4.pdb"
        url = f"https://alphafold.ebi.ac.uk/files/{af_id}"
        response = requests.get(url, stream=True, timeout=60)
    if response and response.status_code == 200:
        with output_path.open("wb") as handle:
            for chunk in response.iter_content(chunk_size=8192):
                handle.write(chunk)
        return output_path
    return None
