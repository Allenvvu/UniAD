#!/usr/bin/env python3
"""
Comprehensive Performance Testing Script for UniAD-ITRI Integration

This script provides enhanced testing capabilities with external GPU monitoring,
detailed metrics collection, and performance validation against targets.

Author: Generated for UniAD Integration Project
"""

import os
import sys
import subprocess
import signal
import time
import json
from pathlib import Path
import threading
from datetime import datetime

# Add paths
sys.path.append('/home/bryan/Desktop/Allen/UniAD/integration')

def start_nvidia_smi_monitoring(output_file: str, interval: int = 1):
    """Start nvidia-smi monitoring in background."""
    cmd = [
        'nvidia-smi',
        '--query-gpu=timestamp,memory.used,memory.total,utilization.gpu,temperature.gpu',
        '--format=csv',
        f'--loop={interval}'
    ]
    
    print(f"🔍 Starting GPU monitoring: {output_file}")
    
    with open(output_file, 'w') as f:
        process = subprocess.Popen(
            cmd, 
            stdout=f, 
            stderr=subprocess.PIPE,
            universal_newlines=True
        )
    
    return process

def stop_nvidia_smi_monitoring(process):
    """Stop nvidia-smi monitoring process."""
    if process and process.poll() is None:
        process.terminate()
        try:
            process.wait(timeout=5)
            print("✅ GPU monitoring stopped")
        except subprocess.TimeoutExpired:
            process.kill()
            print("⚠️  GPU monitoring force killed")

class PerformanceTestSuite:
    """Comprehensive performance test suite."""
    
    def __init__(self, output_dir: str = "output/performance_tests"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)
        self.timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
    def run_cpu_baseline_test(self):
        """Run CPU baseline performance test."""
        print("\n🖥️  CPU Baseline Test")
        print("=" * 50)
        
        gpu_log = self.output_dir / f"gpu_memory_cpu_baseline_{self.timestamp}.csv"
        metrics_file = self.output_dir / f"metrics_cpu_baseline_{self.timestamp}.json"
        
        # Start monitoring
        monitor_process = start_nvidia_smi_monitoring(str(gpu_log))
        
        try:
            # Run CPU test
            cmd = [
                'python', 'integration/phase5_uniad_inference.py',
                '--cpu-debug',
                '--frame', '0',
                '--metrics',
                '--save-metrics', str(metrics_file)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ CPU baseline test completed")
                return True
            else:
                print(f"❌ CPU test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ CPU test timed out (10 minutes)")
            return False
        finally:
            stop_nvidia_smi_monitoring(monitor_process)
            
    def run_gpu_performance_test(self):
        """Run GPU performance test with detailed monitoring."""
        print("\n🚀 GPU Performance Test")
        print("=" * 50)
        
        gpu_log = self.output_dir / f"gpu_memory_performance_{self.timestamp}.csv"
        metrics_file = self.output_dir / f"metrics_gpu_performance_{self.timestamp}.json"
        
        # Start monitoring
        monitor_process = start_nvidia_smi_monitoring(str(gpu_log))
        
        try:
            # Run GPU test
            cmd = [
                'python', 'integration/phase5_uniad_inference.py',
                '--device', 'gpu',
                '--frame', '0',
                '--batch-size', '5',
                '--metrics',
                '--save-metrics', str(metrics_file)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=300)
            
            if result.returncode == 0:
                print("✅ GPU performance test completed")
                return True
            else:
                print(f"❌ GPU test failed: {result.stderr}")
                print(f"STDOUT: {result.stdout}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ GPU test timed out (5 minutes)")
            return False
        finally:
            stop_nvidia_smi_monitoring(monitor_process)
            
    def run_benchmark_test(self):
        """Run comprehensive benchmark test."""
        print("\n🏁 Benchmark Test")
        print("=" * 50)
        
        gpu_log = self.output_dir / f"gpu_memory_benchmark_{self.timestamp}.csv"
        metrics_file = self.output_dir / f"metrics_benchmark_{self.timestamp}.json"
        
        # Start monitoring
        monitor_process = start_nvidia_smi_monitoring(str(gpu_log))
        
        try:
            # Run benchmark
            cmd = [
                'python', 'integration/phase5_uniad_inference.py',
                '--device', 'gpu',
                '--benchmark',
                '--batch-size', '10',
                '--metrics',
                '--save-metrics', str(metrics_file)
            ]
            
            print(f"Running: {' '.join(cmd)}")
            result = subprocess.run(cmd, capture_output=True, text=True, timeout=600)
            
            if result.returncode == 0:
                print("✅ Benchmark test completed")
                return True
            else:
                print(f"❌ Benchmark test failed: {result.stderr}")
                return False
                
        except subprocess.TimeoutExpired:
            print("❌ Benchmark test timed out (10 minutes)")
            return False
        finally:
            stop_nvidia_smi_monitoring(monitor_process)
            
    def analyze_gpu_logs(self, gpu_log_file: str):
        """Analyze GPU memory logs."""
        if not os.path.exists(gpu_log_file):
            return None
            
        try:
            with open(gpu_log_file, 'r') as f:
                lines = f.readlines()[1:]  # Skip header
                
            memory_usage = []
            for line in lines:
                parts = line.strip().split(', ')
                if len(parts) >= 5:
                    try:
                        memory_used = float(parts[1].split(' ')[0])  # Remove 'MiB'
                        memory_usage.append(memory_used / 1024)  # Convert to GB
                    except:
                        continue
                        
            if memory_usage:
                return {
                    'peak_memory_gb': max(memory_usage),
                    'avg_memory_gb': sum(memory_usage) / len(memory_usage),
                    'min_memory_gb': min(memory_usage),
                    'samples': len(memory_usage)
                }
        except Exception as e:
            print(f"⚠️  Error analyzing GPU logs: {e}")
            
        return None
        
    def generate_comprehensive_report(self):
        """Generate comprehensive performance report."""
        report_file = self.output_dir / f"performance_report_{self.timestamp}.md"
        
        print(f"\n📄 Generating performance report: {report_file}")
        
        # Collect all metrics files
        metrics_files = list(self.output_dir.glob(f"metrics_*_{self.timestamp}.json"))
        gpu_logs = list(self.output_dir.glob(f"gpu_memory_*_{self.timestamp}.csv"))
        
        with open(report_file, 'w') as f:
            f.write(f"# UniAD-ITRI Performance Test Report\n\n")
            f.write(f"**Test Date:** {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n")
            f.write(f"**Test Session:** {self.timestamp}\n\n")
            
            f.write("## Performance Targets\n\n")
            f.write("- **Single Frame Inference:** < 200ms\n")
            f.write("- **Batch Throughput:** > 5 FPS\n") 
            f.write("- **Peak Memory Usage:** < 8GB\n\n")
            
            # Analyze each test
            for metrics_file in sorted(metrics_files):
                test_type = metrics_file.stem.replace(f"metrics_", "").replace(f"_{self.timestamp}", "")
                f.write(f"## {test_type.title()} Test Results\n\n")
                
                try:
                    with open(metrics_file, 'r') as mf:
                        data = json.load(mf)
                        summary = data.get('summary', {})
                        
                    f.write(f"### Performance Metrics\n")
                    f.write(f"- **Total Frames:** {summary.get('total_frames', 'N/A')}\n")
                    f.write(f"- **Average Frame Time:** {summary.get('avg_frame_time', 0):.3f}s\n")
                    f.write(f"- **Average FPS:** {summary.get('fps', 0):.2f}\n")
                    f.write(f"- **Peak Memory (Internal):** {summary.get('peak_memory_gb', 0):.2f}GB\n\n")
                    
                    # Performance target analysis
                    f.write(f"### Target Analysis\n")
                    frame_time = summary.get('avg_frame_time', float('inf'))
                    fps = summary.get('fps', 0)
                    peak_memory = summary.get('peak_memory_gb', float('inf'))
                    
                    f.write(f"- **Frame Time Target:** {'✅ PASS' if frame_time <= 0.2 else '❌ FAIL'} ({frame_time:.3f}s)\n")
                    f.write(f"- **FPS Target:** {'✅ PASS' if fps >= 5.0 else '❌ FAIL'} ({fps:.2f} FPS)\n")
                    f.write(f"- **Memory Target:** {'✅ PASS' if peak_memory <= 8.0 else '❌ FAIL'} ({peak_memory:.2f}GB)\n\n")
                    
                except Exception as e:
                    f.write(f"Error loading metrics: {e}\n\n")
                    
                # GPU log analysis
                gpu_log = self.output_dir / f"gpu_memory_{test_type}_{self.timestamp}.csv"
                if gpu_log.exists():
                    gpu_analysis = self.analyze_gpu_logs(str(gpu_log))
                    if gpu_analysis:
                        f.write(f"### External GPU Monitoring\n")
                        f.write(f"- **Peak Memory (nvidia-smi):** {gpu_analysis['peak_memory_gb']:.2f}GB\n")
                        f.write(f"- **Average Memory:** {gpu_analysis['avg_memory_gb']:.2f}GB\n")
                        f.write(f"- **Monitoring Samples:** {gpu_analysis['samples']}\n\n")
                        
            f.write("## Summary\n\n")
            f.write("This report validates the UniAD-ITRI integration performance against defined targets.\n")
            f.write("All tests include comprehensive GPU memory monitoring and detailed metrics collection.\n\n")
            
        print(f"✅ Performance report saved: {report_file}")
        
    def run_full_test_suite(self):
        """Run complete test suite."""
        print("🧪 Starting Comprehensive Performance Test Suite")
        print("=" * 60)
        
        results = {}
        
        # Test 1: CPU Baseline
        results['cpu_baseline'] = self.run_cpu_baseline_test()
        
        # Test 2: GPU Performance
        results['gpu_performance'] = self.run_gpu_performance_test()
        
        # Test 3: Benchmark
        results['benchmark'] = self.run_benchmark_test()
        
        # Generate report
        self.generate_comprehensive_report()
        
        # Overall summary
        print("\n📊 Test Suite Summary")
        print("=" * 30)
        passed = sum(1 for r in results.values() if r)
        total = len(results)
        
        for test_name, passed_test in results.items():
            status = "✅ PASS" if passed_test else "❌ FAIL"
            print(f"   {test_name}: {status}")
            
        print(f"\nOverall: {passed}/{total} tests passed")
        
        if passed == total:
            print("🎉 All performance tests completed successfully!")
            return True
        else:
            print("⚠️  Some performance tests failed - check logs for details")
            return False

def main():
    """Main test execution."""
    import argparse
    
    parser = argparse.ArgumentParser(description='UniAD Performance Test Suite')
    parser.add_argument('--test', choices=['cpu', 'gpu', 'benchmark', 'all'], 
                       default='all', help='Test type to run')
    parser.add_argument('--output-dir', type=str, 
                       default='output/performance_tests',
                       help='Output directory for test results')
    
    args = parser.parse_args()
    
    # Create test suite
    test_suite = PerformanceTestSuite(args.output_dir)
    
    if args.test == 'all':
        success = test_suite.run_full_test_suite()
    elif args.test == 'cpu':
        success = test_suite.run_cpu_baseline_test()
    elif args.test == 'gpu':
        success = test_suite.run_gpu_performance_test()
    elif args.test == 'benchmark':
        success = test_suite.run_benchmark_test()
        
    # Generate individual report if not full suite
    if args.test != 'all':
        test_suite.generate_comprehensive_report()
        
    sys.exit(0 if success else 1)

if __name__ == "__main__":
    main()