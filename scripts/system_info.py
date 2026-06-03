#!/usr/bin/env python3
"""
Affiche les caractéristiques de la machine utilisée pour les expérimentations.
À intégrer dans la section Experimental Setup de l'article.
"""
import platform
import subprocess
import sys

try:
    import psutil
    HAS_PSUTIL = True
except ImportError:
    HAS_PSUTIL = False

try:
    import torch
    HAS_TORCH = True
except ImportError:
    HAS_TORCH = False

def get_cpu_info():
    info = {
        "processor": platform.processor(),
        "machine": platform.machine(),
        "platform": platform.platform(),
        "python_version": sys.version.split()[0],
    }
    if HAS_PSUTIL:
        info["physical_cores"] = psutil.cpu_count(logical=False)
        info["logical_cores"] = psutil.cpu_count(logical=True)
        info["max_freq_mhz"] = psutil.cpu_freq().max if psutil.cpu_freq() else None
        info["ram_gb"] = round(psutil.virtual_memory().total / (1024**3), 2)
    return info

def get_gpu_info():
    gpus = []
    if HAS_TORCH and torch.cuda.is_available():
        for i in range(torch.cuda.device_count()):
            gpus.append({
                "index": i,
                "name": torch.cuda.get_device_name(i),
                "memory_total_gb": round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2),
                "capability": f"{torch.cuda.get_device_properties(i).major}.{torch.cuda.get_device_properties(i).minor}",
            })
    else:
        try:
            out = subprocess.check_output(["nvidia-smi", "--query-gpu=name,memory.total", "--format=csv,noheader,nounits"],
                                          text=True, stderr=subprocess.DEVNULL)
            for i, line in enumerate(out.strip().split("\n")):
                parts = [p.strip() for p in line.split(",")]
                if len(parts) >= 2:
                    gpus.append({
                        "index": i,
                        "name": parts[0],
                        "memory_total_gb": round(float(parts[1]) / 1024, 2) if float(parts[1]) > 1024 else float(parts[1]),
                        "capability": "unknown",
                    })
        except Exception:
            pass
    return gpus

def get_package_versions():
    pkgs = {}
    for pkg in ["numpy", "pandas", "sklearn", "torch", "matplotlib", "seaborn", "tqdm", "psutil"]:
        try:
            mod = __import__(pkg)
            pkgs[pkg] = getattr(mod, "__version__", "unknown")
        except ImportError:
            pkgs[pkg] = "not installed"
    return pkgs

def main():
    print("=" * 60)
    print("SYSTEM INFORMATION")
    print("=" * 60)
    cpu = get_cpu_info()
    for k, v in cpu.items():
        print(f"  {k:20s}: {v}")

    print("\n" + "=" * 60)
    print("GPU INFORMATION")
    print("=" * 60)
    gpus = get_gpu_info()
    if not gpus:
        print("  No GPU detected — running on CPU")
    for g in gpus:
        for k, v in g.items():
            print(f"  {k:20s}: {v}")

    print("\n" + "=" * 60)
    print("PYTHON PACKAGES")
    print("=" * 60)
    pkgs = get_package_versions()
    for k, v in pkgs.items():
        print(f"  {k:20s}: {v}")

    print("\n" + "=" * 60)
    print("LATEX SNIPPET FOR PAPER")
    print("=" * 60)
    latex = rf"""\begin{{itemize}}
    \item CPU: {cpu.get('processor', 'Unknown')} ({cpu.get('logical_cores', '?')} logical cores)
    \item RAM: {cpu.get('ram_gb', '?')} GB
    \item GPU: {gpus[0]['name'] if gpus else 'None'} ({gpus[0]['memory_total_gb'] if gpus else '?'} GB)
    \item OS: {cpu.get('platform', '?')}
    \item Python: {cpu.get('python_version', '?')}
    \item PyTorch: {pkgs.get('torch', '?')}
    \item scikit-learn: {pkgs.get('sklearn', '?')}
\\end{{itemize}}"""
    print(latex)

if __name__ == "__main__":
    main()
