#!/usr/bin/env python3

import os
import sys
import subprocess

def run_script(script_path, description):
    """Run a Python script and handle errors"""
    print(f"\n{'='*60}")
    print(f"Running: {description}")
    print(f"Script: {script_path}")
    print(f"{'='*60}")
    
    try:
        result = subprocess.run([sys.executable, script_path], 
                              capture_output=True, text=True, cwd=os.path.dirname(script_path))
        
        if result.stdout:
            print("STDOUT:")
            print(result.stdout)
        
        if result.stderr:
            print("STDERR:")
            print(result.stderr)
        
        if result.returncode == 0:
            print(f"SUCCESS: {description} completed successfully")
            return True
        else:
            print(f"FAILED: {description} failed with return code {result.returncode}")
            return False
            
    except Exception as e:
        print(f"ERROR: Error running {description}: {e}")
        return False

def main():
    print("="*80)
    print("GENERATING ALL RESULTS FOR SPARSE PRECONDITIONING RESEARCH")
    print("="*80)
    
    # Get the directory where this script is located
    script_dir = os.path.dirname(os.path.abspath(__file__))
    
    # Define all scripts to run
    scripts = [
        (os.path.join(script_dir, "generate_summary_tables.py"), 
         "Summary Performance Tables (ILU0 & ILUK)"),
        
        (os.path.join(script_dir, "correlation_ilu0_updated.py"), 
         "ILU0 Correlation Plot (Wavefront vs Speedup)"),
        
        (os.path.join(script_dir, "correlation_iluk_updated.py"), 
         "ILUK Correlation Plot (Wavefront vs Speedup)"),
        
        (os.path.join(script_dir, "histogram_plots.py"), 
         "Speedup Distribution Histograms (ILU0, ILUK, CPU)"),
        
        (os.path.join(script_dir, "ilu0_application_speedup.py"), 
         "ILU0 Application Speedup Bar Chart"),
        
        (os.path.join(script_dir, "ilu0_factorization_speedup.py"), 
         "ILU0 Factorization Speedup Scatter Plot")
    ]
    
    # Track success/failure
    results = []
    
    # Run each script
    for script_path, description in scripts:
        if os.path.exists(script_path):
            success = run_script(script_path, description)
            results.append((description, success))
        else:
            print(f"ERROR: Script not found: {script_path}")
            results.append((description, False))
    
    # Summary
    print(f"\n{'='*80}")
    print("RESULTS GENERATION SUMMARY")
    print(f"{'='*80}")
    
    successful = 0
    failed = 0
    
    for description, success in results:
        status = "SUCCESS" if success else "FAILED"
        print(f"{status:<10} {description}")
        if success:
            successful += 1
        else:
            failed += 1
    
    print(f"\nTotal: {successful} successful, {failed} failed")
    
    if failed == 0:
        print("\nAll results generated successfully!")
        print("\nOutput locations:")
        print("  Tables: ../../results/performance_summary_tables.txt")
        print("  Plots:  ../../results/plots/")
        
        # List generated files
        results_dir = os.path.join(script_dir, "../../../results")
        plots_dir = os.path.join(results_dir, "plots")
        
        if os.path.exists(results_dir):
            print(f"\nGenerated files:")
            for root, dirs, files in os.walk(results_dir):
                for file in files:
                    rel_path = os.path.relpath(os.path.join(root, file), script_dir)
                    print(f"  - {rel_path}")
    else:
        print(f"\nWarning: {failed} tasks failed. Please check the error messages above.")
        return 1
    
    return 0

if __name__ == "__main__":
    exit(main()) 