"""
Script to test different pruning methods against BTP baseline.

Usage:
    python test_pruning_methods.py --method btp
    python test_pruning_methods.py --method rpd
    python test_pruning_methods.py --method cam
    python test_pruning_methods.py --method hfp

This script helps you configure and test the different pruning methods.
"""

import os
import sys
import argparse
import subprocess
import time


PRUNING_METHODS = {
    'btp': 'Original Balanced Token Pruning (Baseline)',
    'rpd': 'Random Projection Diversity (4× faster diversity)',
    'cam': 'Cross-Attention Mining (free, task-aware)',
    'hfp': 'Hybrid Fast Pruning (combines RPD + CAM)',
    'sgp': 'Spatial Grouped Pruning (2-4× faster, spatial-aware)',
}


def modify_config_in_file(method='btp'):
    """
    Modify the modeling_llama.py to use the specified pruning method.

    This sets the default pruning method in the LlamaModel.__init__.
    """
    file_path = os.path.join(os.path.dirname(__file__), 'modeling_llama.py')

    # Read the file
    with open(file_path, 'r') as f:
        content = f.read()

    # Replace the pruning method line
    import re
    pattern = r"self\.pruning_method = getattr\(config, 'pruning_method', '[a-z]+'\)"
    replacement = f"self.pruning_method = getattr(config, 'pruning_method', '{method}')"

    new_content = re.sub(pattern, replacement, content)

    # Write back
    with open(file_path, 'w') as f:
        f.write(new_content)

    print(f"✓ Set pruning method to: {method}")


def run_benchmark(benchmark='mme', method='btp', gpu=0):
    """
    Run a specific benchmark with the given pruning method.
    """
    print(f"\n{'='*60}")
    print(f"Running {benchmark.upper()} benchmark with {method.upper()}")
    print(f"{'='*60}\n")

    # Benchmark command mapping
    commands = {
        'mme': f'CUDA_VISIBLE_DEVICES={gpu} accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args pretrained="llava-v1.5-7b" --tasks mme --batch_size 1 --log_samples --log_samples_suffix {method}_test --output_path ./logs/',
        'pope': f'CUDA_VISIBLE_DEVICES={gpu} accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args pretrained="llava-v1.5-7b" --tasks pope --batch_size 1 --log_samples --log_samples_suffix {method}_test --output_path ./logs/',
        'gqa': f'CUDA_VISIBLE_DEVICES={gpu} accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args pretrained="llava-v1.5-7b" --tasks gqa --batch_size 1 --log_samples --log_samples_suffix {method}_test --output_path ./logs/',
        'mmbench': f'CUDA_VISIBLE_DEVICES={gpu} accelerate launch --num_processes=1 -m lmms_eval --model llava --model_args pretrained="llava-v1.5-7b" --tasks mmbench_en --batch_size 1 --log_samples --log_samples_suffix {method}_test --output_path ./logs/',
    }

    if benchmark not in commands:
        print(f"Error: Unknown benchmark '{benchmark}'")
        print(f"Available: {list(commands.keys())}")
        return

    # Record start time
    start_time = time.time()

    # Run the command
    cmd = commands[benchmark]
    print(f"Command: {cmd}\n")

    try:
        result = subprocess.run(cmd, shell=True, check=True)
        elapsed = time.time() - start_time

        print(f"\n{'='*60}")
        print(f"✓ Completed in {elapsed:.2f} seconds ({elapsed/60:.2f} minutes)")
        print(f"{'='*60}\n")

        return True
    except subprocess.CalledProcessError as e:
        print(f"\n✗ Benchmark failed with error code {e.returncode}")
        return False


def main():
    parser = argparse.ArgumentParser(
        description='Test different pruning methods',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Available Pruning Methods:
--------------------------
  btp  : Original Balanced Token Pruning (baseline)
  rpd  : Random Projection Diversity (4× faster diversity)
  cam  : Cross-Attention Mining (free, task-aware)
  hfp  : Hybrid Fast Pruning (combines RPD + CAM)
  sgp  : Spatial Grouped Pruning (2-4× faster, spatial-aware)

Examples:
---------
  # Test RPD on MME benchmark
  python test_pruning_methods.py --method rpd --benchmark mme

  # Test all methods on POPE
  python test_pruning_methods.py --method all --benchmark pope

  # Quick comparison: BTP vs HFP on MME
  python test_pruning_methods.py --method btp hfp --benchmark mme
        """
    )

    parser.add_argument(
        '--method',
        type=str,
        nargs='+',
        default=['btp'],
        choices=list(PRUNING_METHODS.keys()) + ['all'],
        help='Pruning method(s) to test'
    )

    parser.add_argument(
        '--benchmark',
        type=str,
        default='mme',
        choices=['mme', 'pope', 'gqa', 'mmbench', 'all'],
        help='Benchmark to run'
    )

    parser.add_argument(
        '--gpu',
        type=int,
        default=0,
        help='GPU device ID'
    )

    parser.add_argument(
        '--configure-only',
        action='store_true',
        help='Only configure the method, do not run benchmark'
    )

    args = parser.parse_args()

    # Handle 'all' methods
    if 'all' in args.method:
        methods = list(PRUNING_METHODS.keys())
    else:
        methods = args.method

    # Handle 'all' benchmarks
    if args.benchmark == 'all':
        benchmarks = ['mme', 'pope', 'gqa', 'mmbench']
    else:
        benchmarks = [args.benchmark]

    print("\n" + "="*60)
    print("Pruning Method Testing Framework")
    print("="*60)
    print(f"\nMethods to test: {', '.join(methods)}")
    print(f"Benchmarks: {', '.join(benchmarks)}")
    print(f"GPU: {args.gpu}\n")

    # Run tests for each method and benchmark
    results = {}

    for method in methods:
        print(f"\n{'#'*60}")
        print(f"Testing method: {method.upper()}")
        print(f"Description: {PRUNING_METHODS[method]}")
        print(f"{'#'*60}\n")

        # Configure the method
        modify_config_in_file(method)

        if args.configure_only:
            print(f"✓ Configured {method}. Skipping benchmark run.")
            continue

        results[method] = {}

        for benchmark in benchmarks:
            success = run_benchmark(benchmark, method, args.gpu)
            results[method][benchmark] = success

            if not success:
                print(f"Warning: {benchmark} failed for {method}")

    # Print summary
    if not args.configure_only:
        print("\n" + "="*60)
        print("SUMMARY")
        print("="*60 + "\n")

        for method in methods:
            print(f"\n{method.upper()}:")
            for benchmark in benchmarks:
                status = "✓" if results[method].get(benchmark, False) else "✗"
                print(f"  {status} {benchmark}")

        print("\nResults saved to: ./logs/")
        print("\nTo analyze results, check:")
        print("  - ./logs/*_{method}_test/")


if __name__ == '__main__':
    main()
