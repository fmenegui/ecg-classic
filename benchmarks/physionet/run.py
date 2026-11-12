#!/usr/bin/env python3
"""Benchmark no dataset do Kaggle "PhysioNet - Digitization of ECG Images".

  inspect : lista os arquivos da competição e baixa um registro de exemplo, para conferir
            o formato (nomes, colunas, frequência, duração por derivação).
  eval    : (depois de conferido o formato) baixa N registros do treino, digitaliza cada
            imagem com o ecgdig e mede SNR contra o sinal verdadeiro.

Autenticação: variável KAGGLE_API_TOKEN (secret do repositório), CLI kaggle>=2.2.
"""
from __future__ import annotations

import argparse
import csv
import io
import json
import os
import re
import subprocess
import sys
import zipfile
from collections import Counter
from pathlib import Path

COMP = os.environ.get("COMP", "physionet-ecg-image-digitization")
HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE.parents[1]))


def kaggle(*args, check=True):
    r = subprocess.run(["kaggle", *args], capture_output=True, text=True)
    if check and r.returncode != 0:
        raise RuntimeError(f"kaggle {' '.join(args)}\n{r.stdout}\n{r.stderr}")
    return r.stdout


def list_files(max_pages=400):
    """Todos os arquivos (nome, tamanho) via CLI paginada."""
    out, token = [], None
    for _ in range(max_pages):
        args = ["competitions", "files", COMP, "--page-size", "200", "-v"]
        if token:
            args += ["--page-token", token]
        txt = kaggle(*args)
        lines = [l for l in txt.splitlines() if l.strip()]
        token = None
        rows = []
        for l in lines:
            m = re.match(r"Next Page Token = (\S+)", l)
            if m:
                token = m.group(1)
            elif "," in l:
                rows.append(l)
        rd = csv.reader(io.StringIO("\n".join(rows)))
        hdr = None
        for r in rd:
            if hdr is None:
                hdr = r
                continue
            out.append(dict(zip(hdr, r)))
        if not token:
            break
    return out


def download(name, dest: Path):
    """Baixa um arquivo da competição; devolve o caminho local (descompacta se vier .zip)."""
    dest.mkdir(parents=True, exist_ok=True)
    before = {p for p in dest.rglob("*")}
    log = kaggle("competitions", "download", COMP, "-f", name, "-p", str(dest), check=False)
    new = [p for p in dest.rglob("*") if p not in before and p.is_file()]
    for z in [p for p in new if p.suffix == ".zip"]:
        try:
            with zipfile.ZipFile(z) as zf:
                zf.extractall(dest)
            z.unlink()
        except zipfile.BadZipFile:
            pass
    base = Path(name).name
    hits = [p for p in dest.rglob("*") if p.is_file() and p.name == base]
    if not hits:
        raise FileNotFoundError(f"{name} não apareceu em {dest}; saída do CLI: {log.strip()[:300]}")
    return hits[0]


def inspect(out: Path):
    out.mkdir(parents=True, exist_ok=True)
    files = list_files()
    (out / "files.json").write_text(json.dumps(files, indent=0))
    names = [f.get("name") or f.get("ref") or "" for f in files]
    L = ["# PhysioNet (Kaggle): inspeção", "", f"{len(names)} arquivos.", ""]
    top = Counter(n.split("/")[0] for n in names)
    ext = Counter(Path(n).suffix for n in names)
    L += ["Diretórios de topo: " + ", ".join(f"{k} ({v})" for k, v in top.most_common(20)), "",
          "Extensões: " + ", ".join(f"{k or '∅'} ({v})" for k, v in ext.most_common(20)), "",
          "Primeiros 60 nomes:", "", "```", *names[:60], "```", ""]
    # CSVs pequenos de topo e um registro de exemplo
    small = [f for f in files if "/" not in (f.get("name") or "") and (f.get("name") or "").endswith(".csv")]
    for f in small[:6]:
        try:
            p = download(f["name"], out / "sample")
            head = p.read_text(errors="replace").splitlines()[:6]
        except Exception as e:  # noqa: BLE001
            head = [repr(e)]
        L += [f"## {f['name']} ({f.get('size', '?')})", "", "```", *head, "```", ""]
    # primeiro registro de treino (todos os arquivos com o mesmo prefixo de pasta)
    train = [n for n in names if n.startswith("train/")]
    if train:
        rec = "/".join(train[0].split("/")[:2])
        members = [n for n in names if n.startswith(rec + "/") or n.startswith(rec + "-") or n == rec]
        L += [f"## Registro de exemplo `{rec}` ({len(members)} arquivos)", "", "```", *members[:40], "```", ""]
        for n in members[:20]:
            try:
                p = download(n, out / "sample" / Path(n).parent)
            except Exception as e:  # noqa: BLE001
                L.append(f"falhou {n}: {e}")
                continue
            if p.suffix == ".csv":
                txt = p.read_text(errors="replace").splitlines()
                (out / "sample_record.csv").write_text("\n".join(txt[:20]))
                L += [f"`{n}`: {len(txt)} linhas", "", "```", *txt[:4], "```", ""]
            elif p.suffix.lower() in (".png", ".jpg", ".jpeg"):
                import cv2
                im = cv2.imread(str(p))
                L.append(f"`{n}`: imagem {None if im is None else im.shape}")
        L.append("")
    L += ["## Arquivos baixados", "", "```", *[str(p) for p in sorted((out / "sample").rglob("*"))][:80], "```"]
    (out / "report.md").write_text("\n".join(L))
    print("\n".join(L))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("mode", choices=["inspect", "eval"])
    ap.add_argument("--out", type=Path, default=Path("results/physionet"))
    ap.add_argument("--n", type=int, default=40)
    ap.add_argument("--config", action="append", default=[])
    a = ap.parse_args()
    if a.mode == "inspect":
        inspect(a.out)
    else:
        from eval_snr import evaluate  # escrito depois da inspeção do formato
        evaluate(a)


if __name__ == "__main__":
    sys.path.insert(0, str(HERE))
    main()
