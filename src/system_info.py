"""
src/system_info.py

Hardware and software environment discovery script.
Reports OS, Python version, CPU specifications, RAM capacity, GPU/CUDA availability,
and installed package versions to determine suitable local LLM execution capabilities.
"""

import sys
import os
import platform
import subprocess
import torch
import sentence_transformers


def get_cpu_info():
    cpu_name = platform.processor() or "Unknown CPU"
    if platform.system() == "Windows":
        try:
            cmd = "powershell -Command \"(Get-CimInstance Win32_Processor).Name\""
            res = subprocess.check_output(cmd, shell=True).decode().strip()
            if res:
                cpu_name = res
        except Exception:
            pass
    return cpu_name


def get_ram_info():
    total_ram = "Unknown"
    avail_ram = "Unknown"
    
    # Try using psutil if available, otherwise use Windows CIM/WMI
    try:
        import psutil
        mem = psutil.virtual_memory()
        total_ram = f"{mem.total / (1024**3):.2f} GB"
        avail_ram = f"{mem.available / (1024**3):.2f} GB"
        return total_ram, avail_ram
    except ImportError:
        pass

    if platform.system() == "Windows":
        try:
            cmd_total = "powershell -Command \"[math]::round((Get-CimInstance Win32_ComputerSystem).TotalPhysicalMemory / 1GB, 2)\""
            res_total = subprocess.check_output(cmd_total, shell=True).decode().strip()
            if res_total:
                total_ram = f"{res_total} GB"
        except Exception:
            pass

        try:
            cmd_free = "powershell -Command \"[math]::round((Get-CimInstance Win32_OperatingSystem).FreePhysicalMemory / 1MB, 2)\""
            res_free = subprocess.check_output(cmd_free, shell=True).decode().strip()
            if res_free:
                avail_ram = f"{res_free} GB"
        except Exception:
            pass

    return total_ram, avail_ram


def run_system_check():
    print("==================================================")
    print("       SYSTEM HARDWARE & SOFTWARE REPORT          ")
    print("==================================================")

    # 1. OS Info
    os_info = f"{platform.system()} {platform.release()} ({platform.architecture()[0]})"
    print(f"1. Operating System    : {os_info}")

    # 2. Python Version
    print(f"2. Python Version       : {sys.version.split()[0]} ({sys.executable})")

    # 3 & 4. CPU Info
    cpu_name = get_cpu_info()
    cpu_cores = os.cpu_count()
    print(f"3. CPU Model/Name       : {cpu_name}")
    print(f"4. CPU Logical Cores    : {cpu_cores}")

    # 5 & 6. RAM Info
    total_ram, avail_ram = get_ram_info()
    print(f"5. Total RAM            : {total_ram}")
    print(f"6. Available RAM        : {avail_ram}")

    # 7 & 8 & 9. GPU & CUDA
    cuda_available = torch.cuda.is_available()
    print(f"7. NVIDIA GPU Available : {'YES' if cuda_available else 'NO (or CPU PyTorch)'}")

    if cuda_available:
        gpu_count = torch.cuda.device_count()
        print(f"   GPU Count            : {gpu_count}")
        for i in range(gpu_count):
            gpu_name = torch.cuda.get_device_name(i)
            vram_bytes = torch.cuda.get_device_properties(i).total_memory
            vram_gb = f"{vram_bytes / (1024**3):.2f} GB"
            print(f"8. GPU #{i} Details     : {gpu_name} (VRAM: {vram_gb})")
    else:
        # Check system for nvidia-smi if PyTorch is built CPU-only
        try:
            smi = subprocess.check_output("nvidia-smi --query-gpu=name,memory.total --format=csv,noheader", shell=True).decode().strip()
            if smi:
                print(f"8. GPU Hardware Found   : {smi} (PyTorch is CPU build)")
        except Exception:
            print("8. GPU Details          : None detected")

    print(f"9. CUDA Enabled PyTorch : {'YES' if cuda_available else 'NO'}")

    # 10. Installed PyTorch Version
    print(f"10. PyTorch Version     : {torch.__version__}")

    # 11. Installed Sentence Transformers Version
    print(f"11. Sentence-Transformers: {sentence_transformers.__version__}")
    print("==================================================")


if __name__ == "__main__":
    run_system_check()
