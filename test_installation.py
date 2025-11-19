#!/usr/bin/env python3
"""Test script to verify input-representation-benchmark installation.

This script checks that all components are properly installed and can import
both input_representation and ethos modules correctly.

Usage:
    python test_installation.py
"""

import sys
from pathlib import Path


def print_test(name: str, passed: bool, details: str = ""):
    """Print test result with formatting."""
    status = "✓ PASS" if passed else "✗ FAIL"
    print(f"{status:8} | {name}")
    if details and not passed:
        print(f"         | Details: {details}")


def test_input_representation_import():
    """Test that input_representation package can be imported."""
    try:
        import input_representation
        version = getattr(input_representation, "__version__", "unknown")
        print_test("Import input_representation", True, f"version {version}")
        return True
    except ImportError as e:
        print_test("Import input_representation", False, str(e))
        return False


def test_ethos_import():
    """Test that ethos package can be imported."""
    try:
        import ethos
        print_test("Import ethos", True)
        return True
    except ImportError as e:
        print_test("Import ethos", False, str(e))
        print("         | Hint: Install ethos-ares with: pip install -e /path/to/ethos-ares")
        return False


def test_quantization_module():
    """Test quantization module imports."""
    try:
        from input_representation.tokenize.common import QuantizationVentile, transform_to_ventiles
        print_test("Import QuantizationVentile", True)
        return True
    except ImportError as e:
        print_test("Import QuantizationVentile", False, str(e))
        return False


def test_ethos_components():
    """Test importing ethos components directly."""
    try:
        from ethos.tokenize.patterns import MatchAndRevise
        print_test("Import MatchAndRevise directly", True)
        return True
    except ImportError:
        # Fallback for development environment where ethos might not be installed but in path
        try:
            import sys
            from pathlib import Path
            # Try to find ethos relative to this script
            current_dir = Path(__file__).parent
            ethos_path = current_dir.parent / "ethos-ares" / "src"
            if ethos_path.exists() and str(ethos_path) not in sys.path:
                sys.path.insert(0, str(ethos_path))
            
            from ethos.tokenize.patterns import MatchAndRevise
            print_test("Import MatchAndRevise (dev mode)", True)
            return True
        except Exception as e:
            print_test("Import MatchAndRevise", False, str(e))
            return False


def test_configs():
    """Test configuration files exist."""
    try:
        # Current structure: benchmarks/ventile-quantization/configs
        benchmarks_dir = Path(__file__).parent / "benchmarks" / "ventile-quantization"
        config_path = benchmarks_dir / "configs" / "event_configs_v3.1.yaml"
        
        configs_exist = config_path.exists()
        print_test("Configuration files exist", configs_exist)
        return configs_exist
    except Exception as e:
        print_test("Configuration files exist", False, str(e))
        return False


def test_scripts():
    """Test that experiment scripts directory exists with required scripts."""
    try:
        # scripts moved to benchmarks/ventile-quantization/scripts
        scripts_dir = Path(__file__).parent / "benchmarks" / "ventile-quantization" / "scripts"
        required_scripts = [
            "02_run_ventile_quantization.py",
            "03_analyze_ventiles.py",
        ]
        
        all_exist = all((scripts_dir / script).exists() for script in required_scripts)
        print_test("Benchmark scripts exist", all_exist)
        return all_exist
    except Exception as e:
        print_test("Benchmark scripts exist", False, str(e))
        return False


def test_ventile_computation():
    """Test basic ventile computation."""
    try:
        import numpy as np
        from input_representation.tokenize.common.quantization import QuantizationVentile
        
        quantizator = QuantizationVentile
        values = np.random.normal(100, 20, 1000).tolist()
        
        # Test with no reference ranges
        breaks = quantizator._compute_breaks_no_ranges(np.array(values), num_bins=20)
        has_19_breaks = len(breaks) == 19  # 20 bins = 19 breaks
        
        # Test with both reference ranges
        breaks_ref = quantizator._compute_breaks_both_ranges(
            np.array(values), ref_lower=80, ref_upper=120, num_bins=20
        )
        has_ref_bounds = 80 in breaks_ref and 120 in breaks_ref
        
        passed = has_19_breaks and has_ref_bounds
        print_test("Ventile computation logic", passed)
        return passed
    except Exception as e:
        print_test("Ventile computation logic", False, str(e))
        return False


def test_dependencies():
    """Test required dependencies."""
    required_packages = [
        ("polars", "Polars"),
        ("numpy", "NumPy"),
        ("hydra", "Hydra"),
        ("loguru", "Loguru"),
    ]
    
    all_installed = True
    for module_name, display_name in required_packages:
        try:
            __import__(module_name)
            print_test(f"Dependency: {display_name}", True)
        except ImportError:
            print_test(f"Dependency: {display_name}", False, f"{display_name} not installed")
            all_installed = False
    
    return all_installed


def main():
    """Run all tests."""
    print("=" * 60)
    print("Input Representation Benchmark - Installation Test")
    print("=" * 60)
    print()
    
    tests = [
        ("Core Packages", [
            test_input_representation_import,
            test_ethos_import,
        ]),
        ("Dependencies", [
            test_dependencies,
        ]),
        ("Modules", [
            test_quantization_module,
        ]),
        ("Integration", [
            test_ethos_components,
            test_configs,
            test_scripts,
        ]),
        ("Functionality", [
            test_ventile_computation,
        ]),
    ]
    
    all_passed = True
    
    for category, category_tests in tests:
        print(f"\n{category}:")
        print("-" * 60)
        
        for test_func in category_tests:
            passed = test_func()
            all_passed = all_passed and passed
    
    print("\n" + "=" * 60)
    if all_passed:
        print("✓ ALL TESTS PASSED")
        print("\nYour installation is ready to use!")
        print("\nNext steps:")
        print("  1. Read QUICKSTART.md for usage instructions")
        print("  2. Run tokenization on your MIMIC-IV MEDS data")
        print("  3. Train a model with ethos_train")
    else:
        print("✗ SOME TESTS FAILED")
        print("\nPlease fix the failed tests before proceeding.")
        print("Common issues:")
        print("  - ethos-ares not installed: pip install -e /path/to/ethos-ares")
        print("  - Missing dependencies: pip install -e .")
        print("  - Wrong Python version: Requires Python 3.12+")
    print("=" * 60)
    
    return 0 if all_passed else 1


if __name__ == "__main__":
    sys.exit(main())

