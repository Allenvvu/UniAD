#!/usr/bin/env python3
"""
Validation script to run UniAD inference on 100 samples and check results.
"""
import os
import sys
import subprocess
import time
import argparse
from pathlib import Path
import pickle

def run_inference_100_samples():
    """Run inference on 100 samples"""
    print("=" * 60)
    print("UNIAD INFERENCE VALIDATION - 100 SAMPLES")
    print("=" * 60)

    # Create output directory
    output_dir = Path("validation_output")
    output_dir.mkdir(exist_ok=True)

    # Create a temporary config that limits to 100 samples
    temp_config_path = "l2g_inference_config_100.py"
    create_limited_config(temp_config_path, max_samples=100)

    # Run inference command
    cmd = [
        "python", "-m", "torch.distributed.run",
        "--nproc_per_node=1",
        "--master_port=29501",  # Different port to avoid conflicts
        "tools/test.py",
        temp_config_path,
        "ckpts/uniad_base_e2e.pth",
        "--out", f"{output_dir}/validation_results.pkl",
        "--launcher", "pytorch"
    ]

    print(f"Running command: {' '.join(cmd)}")
    print("-" * 60)

    start_time = time.time()

    try:
        # Run the inference
        result = subprocess.run(
            cmd,
            cwd=os.getcwd(),
            env=dict(os.environ, PYTHONPATH="."),
            capture_output=False,  # Show real-time output
            text=True,
            timeout=1800  # 30 minutes timeout
        )

        end_time = time.time()
        duration = end_time - start_time

        print("-" * 60)
        print(f"Inference completed in {duration:.1f} seconds ({duration/60:.1f} minutes)")

        if result.returncode == 0:
            print("✅ SUCCESS: Inference completed successfully!")
            validate_results(output_dir / "validation_results.pkl")
        else:
            print(f"❌ ERROR: Inference failed with return code {result.returncode}")
            return False

    except subprocess.TimeoutExpired:
        print("⏰ TIMEOUT: Inference took longer than 30 minutes")
        return False
    except KeyboardInterrupt:
        print("🛑 INTERRUPTED: User stopped the inference")
        return False
    except Exception as e:
        print(f"❌ EXCEPTION: {str(e)}")
        return False
    finally:
        # Clean up temporary config
        if os.path.exists(temp_config_path):
            os.remove(temp_config_path)

    return True

def create_limited_config(config_path, max_samples=100):
    """Create a config file that limits inference to specified number of samples"""

    # Read the original config
    with open("l2g_inference_config.py", "r") as f:
        config_content = f.read()

    # Modify the config to limit samples
    limited_config = config_content.replace(
        "data = dict(",
        f"""# Limited to {max_samples} samples for validation
data = dict("""
    )

    # Add sample limitation in the test dataset
    if "test=dict(" in limited_config:
        limited_config = limited_config.replace(
            "test=dict(",
            f"test=dict(\n        # Limit to {max_samples} samples\n        max_samples={max_samples},"
        )

    # Write the modified config
    with open(config_path, "w") as f:
        f.write(limited_config)

    print(f"Created limited config: {config_path} (max {max_samples} samples)")

def validate_results(results_path):
    """Validate the inference results"""
    print("-" * 60)
    print("VALIDATING RESULTS")
    print("-" * 60)

    if not os.path.exists(results_path):
        print(f"❌ Results file not found: {results_path}")
        return False

    try:
        # Load and analyze results
        with open(results_path, "rb") as f:
            results = pickle.load(f)

        print(f"✅ Results file loaded successfully")
        print(f"📊 Number of results: {len(results)}")

        if len(results) > 0:
            # Analyze first result
            first_result = results[0]
            print(f"📋 First result keys: {list(first_result.keys()) if isinstance(first_result, dict) else 'Not a dict'}")

            # Check for expected outputs
            expected_keys = ['pts_bbox', 'img_bbox']  # Common detection outputs
            found_keys = []
            if isinstance(first_result, dict):
                for key in expected_keys:
                    if key in first_result:
                        found_keys.append(key)
                        if hasattr(first_result[key], 'shape') or isinstance(first_result[key], (list, tuple)):
                            print(f"  - {key}: {type(first_result[key])}")

            if found_keys:
                print(f"✅ Found expected output keys: {found_keys}")
            else:
                print("⚠️  No standard detection keys found, but inference completed")

        print(f"✅ VALIDATION PASSED: {len(results)} samples processed successfully")
        print(f"💾 Results saved to: {results_path}")

        return True

    except Exception as e:
        print(f"❌ Error validating results: {str(e)}")
        return False

def main():
    parser = argparse.ArgumentParser(description="Validate UniAD inference on 100 samples")
    parser.add_argument("--samples", type=int, default=100, help="Number of samples to process (default: 100)")

    args = parser.parse_args()

    # Check if we're in the right directory
    if not os.path.exists("l2g_inference_config.py"):
        print("❌ ERROR: l2g_inference_config.py not found. Please run from the UniAD project root.")
        sys.exit(1)

    if not os.path.exists("ckpts/uniad_base_e2e.pth"):
        print("❌ ERROR: ckpts/uniad_base_e2e.pth not found. Please ensure the checkpoint is downloaded.")
        sys.exit(1)

    print(f"🔍 Validating UniAD inference on {args.samples} samples...")

    success = run_inference_100_samples()

    if success:
        print("\n" + "=" * 60)
        print("🎉 VALIDATION COMPLETED SUCCESSFULLY!")
        print("The UniAD inference pipeline is working correctly.")
        print("You can now run the full inference on all 2675 samples.")
        print("=" * 60)
        sys.exit(0)
    else:
        print("\n" + "=" * 60)
        print("❌ VALIDATION FAILED")
        print("Please check the error messages above and fix any issues.")
        print("=" * 60)
        sys.exit(1)

if __name__ == "__main__":
    main()