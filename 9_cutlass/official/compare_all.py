"""
Comprehensive comparison of all official CUTLASS GEMM implementations.

This script serves as a master benchmark runner, executing a series of GEMM (General
Matrix-Matrix Multiplication) benchmarks to compare the performance of different
CUTLASS implementations provided in the `official` directory.

It is designed to be run from the `book.cu/9_cutlass/official` directory and will
sequentially execute the `benchmark.py` script found in each of the specified
subdirectories.

The primary purpose is to provide a side-by-side view of:
1.  **Ampere vs. Hopper**: Compares the performance of GEMM kernels optimized for the
    NVIDIA Ampere (e.g., A100) and Hopper (e.g., H100) architectures.
2.  **Single-GPU vs. Multi-GPU**: Demonstrates the performance of both single-GPU kernels
    and the distributed multi-GPU implementation.

After running all benchmarks, it prints a summary of the results and key takeaways
about the performance differences and the underlying CUTLASS features that contribute
to them.
"""
import subprocess
import sys
import os

def run_benchmark(name: str, directory: str):
    """
    Runs a specific benchmark in a given subdirectory.

    This function changes the current working directory to the specified benchmark
    directory and executes the `benchmark.py` script within it. It captures the
    output and checks for errors.

    Args:
        name (str): The human-readable name of the benchmark for display purposes.
        directory (str): The subdirectory containing the benchmark to run.

    Returns:
        bool: True if the benchmark ran successfully, False otherwise.
    """
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"Directory: {directory}")
    print(f"{'='*70}\n")
    
    try:
        # Execute the benchmark.py script in the specified directory.
        # `check=True` will raise a CalledProcessError if the script returns a non-zero exit code.
        # `text=True` ensures stdout and stderr are decoded as text.
        subprocess.run(
            [sys.executable, "benchmark.py"], # Use sys.executable to ensure the same python interpreter is used
            cwd=directory,
            capture_output=False, # We want the output to be printed to the console directly
            text=True,
            check=True
        )
        return True
    except FileNotFoundError:
        print(f"❌ {name} failed: benchmark.py not found in {directory}")
        return False
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} failed with error code {e.returncode}")
        # The benchmark script's output will already be on the console.
        return False

def main():
    """
    Main function to orchestrate the benchmark runs and display the summary.
    """
    print("""
    ╔═══════════════════════════════════════════════════════════════════╗
    ║                                                                   ║
    ║        Official CUTLASS GEMM Implementations - Comparison         ║
    ║                                                                   ║
    ║   This script will run benchmarks for Ampere, Hopper, and Multi-GPU ║
    ║   kernels to compare their performance characteristics.             ║
    ║                                                                   ║
    ╚═══════════════════════════════════════════════════════════════════╝
    """)
    
    # List of benchmarks to run. Each entry is a tuple of (name, directory).
    # Assumes this script is run from the `book.cu/9_cutlass/official` directory.
    benchmarks = [
        ("⚡ Official Ampere GEMM (SM80)", "official_ampere_gemm"),
        ("⚡ Official Hopper GEMM (SM90, Single-GPU)", "official_hopper_gemm"),
        ("🔗 Official Multi-GPU GEMM (SM90, Distributed)", "official_multi_gpu_gemm"),
    ]
    
    results = {}
    
    for name, directory in benchmarks:
        # Check if the benchmark directory exists before trying to run it.
        if not os.path.isdir(directory):
            print(f"\n⚠️  Skipping: {name} (Directory not found: {directory})")
            results[name] = "❔"
            continue
        
        success = run_benchmark(name, directory)
        results[name] = "✅" if success else "❌"
    
    
    print("\n" + "="*70)
    print("📊 BENCHMARK SUMMARY")
    print("="*70)
    
    for name, status in results.items():
        print(f"{status} {name}")
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                        KEY TAKEAWAYS                              ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  1. Performance varies significantly by architecture. Hopper (SM90)  ║
║     kernels using TMA (Tensor Memory Accelerator) and other features  ║
║     outperform Ampere (SM80) kernels.                               ║
║                                                                   ║
║  2. The official examples are highly tuned but may not match cuBLAS. ║
║     → For production, use the CUTLASS Profiler to find the optimal  ║
║       tile sizes and configurations for your specific problem.      ║
║                                                                   ║
║  3. Key optimizations in modern CUTLASS (Hopper):                  ║
║     • Larger tile sizes (e.g., 128x256) and cluster shapes.         ║
║     • Asynchronous data movement with TMA.                          ║
║     • Specialized kernel schedules (`KernelTmaWarpSpecialized...`). ║
║                                                                   ║
║  4. The multi-GPU example demonstrates true distributed GEMM.        ║
║     → It solves a single large problem across GPUs, involving       ║
║       communication, unlike simple data-parallel approaches.        ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
    """)

if __name__ == "__main__":
    # Set the working directory to the script's own directory to ensure
    # relative paths to benchmark directories are correct.
    os.chdir(os.path.dirname(os.path.abspath(__file__)))
    main()

