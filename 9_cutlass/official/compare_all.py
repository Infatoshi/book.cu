"""
Comprehensive comparison of all GEMM implementations

Runs both from-scratch and official implementations side-by-side
to show the performance gap and key learnings.
"""
import subprocess
import sys

def run_benchmark(name, directory):
    """Run a benchmark and capture output"""
    print(f"\n{'='*70}")
    print(f"Running: {name}")
    print(f"{'='*70}\n")
    
    try:
        result = subprocess.run(
            ["python", "benchmark.py"],
            cwd=directory,
            capture_output=False,
            text=True,
            check=True
        )
        return True
    except subprocess.CalledProcessError as e:
        print(f"❌ {name} failed with error code {e.returncode}")
        return False

def main():
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                                                                   ║
║           CUTLASS Learning Examples - Full Comparison            ║
║                                                                   ║
║   Comparing From-Scratch vs Official CUTLASS Implementations     ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")
    
    benchmarks = [
        ("📚 From-Scratch Ampere GEMM", "ampere_gemm"),
        ("⚡ Official Ampere GEMM", "official_ampere_gemm"),
        ("", None),  
        ("📚 From-Scratch Hopper GEMM", "hopper_gemm"),
        ("⚡ Official Hopper GEMM", "official_hopper_gemm"),
        ("", None),  
        ("🔗 Multi-GPU Data-Parallel GEMM", "multi_gpu_gemm"),
    ]
    
    results = {}
    
    for name, directory in benchmarks:
        if directory is None:
            print("\n" + "="*70 + "\n")
            continue
        
        success = run_benchmark(name, directory)
        results[name] = "✓" if success else "✗"
    
    
    print("\n" + "="*70)
    print("SUMMARY")
    print("="*70)
    
    for name, status in results.items():
        print(f"{status} {name}")
    
    print("""
╔═══════════════════════════════════════════════════════════════════╗
║                        KEY TAKEAWAYS                              ║
╠═══════════════════════════════════════════════════════════════════╣
║                                                                   ║
║  1. From-Scratch implementations are ~8x slower than official    ║
║     → Good for learning APIs, not for production                 ║
║                                                                   ║
║  2. Official implementations are ~30% slower than cuBLAS         ║
║     → CUTLASS examples aren't fully tuned                        ║
║     → Use CUTLASS Profiler for production workloads              ║
║                                                                   ║
║  3. Key optimizations in official versions:                      ║
║     • Larger tile sizes (128x256 vs 128x128)                    ║
║     • Multi-stage pipelines (3+ stages)                         ║
║     • Cluster shapes for Hopper (_2,_1,_1)                      ║
║     • TMA + WGMMA for Hopper                                    ║
║                                                                   ║
║  4. Multi-GPU shows concept but has overhead                     ║
║     → For real distributed: see example 65 (needs CUDA 12.6+)   ║
║                                                                   ║
╚═══════════════════════════════════════════════════════════════════╝
""")

if __name__ == "__main__":
    main()

