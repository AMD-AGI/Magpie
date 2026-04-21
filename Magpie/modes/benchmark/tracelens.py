###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
TraceLens integration for benchmark mode.

Provides trace analysis and CSV export functionality using TraceLens CLI.

CLI Commands used:
- TraceLens_generate_perf_report_pytorch: Single trace analysis
- TraceLens_generate_multi_rank_collective_report_pytorch: Multi-rank collective analysis
- TraceLens_compare_perf_reports_pytorch: Compare multiple performance reports
"""

import logging
import shutil
import subprocess
import sys
from pathlib import Path
from typing import Dict, List, Optional, Any

from .config import TraceLensConfig

logger = logging.getLogger(__name__)

# TraceLens installation URL
TRACELENS_INSTALL_URL = "git+https://github.com/AMD-AIG-AIMA/TraceLens.git"

# CLI command names
CLI_GENERATE_REPORT = "TraceLens_generate_perf_report_pytorch"
CLI_MULTI_RANK_COLLECTIVE = "TraceLens_generate_multi_rank_collective_report_pytorch"
CLI_COMPARE_REPORTS = "TraceLens_compare_perf_reports_pytorch"


def ensure_tracelens_installed() -> bool:
    """
    Check if TraceLens is installed, and install it if not.
    
    Returns:
        True if TraceLens is available (installed or just installed)
    """
    # Check if CLI command exists
    if shutil.which(CLI_GENERATE_REPORT):
        logger.debug("TraceLens CLI is already available")
        return True
    
    # Try importing as fallback check
    try:
        __import__("TraceLens")
        logger.debug("TraceLens is already installed")
        return True
    except ImportError:
        pass
    
    logger.info(f"TraceLens not found. Installing from {TRACELENS_INSTALL_URL}...")
    
    try:
        # Install TraceLens using pip
        result = subprocess.run(
            [sys.executable, "-m", "pip", "install", TRACELENS_INSTALL_URL],
            capture_output=True,
            text=True,
            timeout=300,  # 5 minute timeout for installation
        )
        
        if result.returncode == 0:
            logger.info("TraceLens installed successfully")
            return True
        else:
            logger.error(f"Failed to install TraceLens: {result.stderr}")
            return False
            
    except subprocess.TimeoutExpired:
        logger.error("TraceLens installation timed out")
        return False
    except Exception as e:
        logger.error(f"Failed to install TraceLens: {e}")
        return False


class TraceLensAnalyzer:
    """
    TraceLens trace analyzer for torch profiler traces.
    
    Uses TraceLens CLI commands for analysis:
    - TraceLens_generate_perf_report_pytorch: Single trace analysis
    - TraceLens_generate_multi_rank_collective_report_pytorch: Multi-rank collective
    - TraceLens_compare_perf_reports_pytorch: Compare reports
    """
    
    def __init__(self, config: TraceLensConfig):
        """
        Initialize TraceLens analyzer.
        
        Args:
            config: TraceLens configuration
        """
        self.config = config
        self._tracelens_available: Optional[bool] = None
    
    def is_available(self) -> bool:
        """
        Check if TraceLens CLI is installed and available.
        Will attempt to install if not found.
        
        Returns:
            True if TraceLens is available
        """
        if self._tracelens_available is not None:
            return self._tracelens_available
        
        # First, try to ensure TraceLens is installed
        if not ensure_tracelens_installed():
            self._tracelens_available = False
            return False
        
        # Verify CLI command exists
        if shutil.which(CLI_GENERATE_REPORT):
            self._tracelens_available = True
            logger.info("TraceLens CLI is available")
        else:
            logger.error(f"TraceLens installed but CLI command '{CLI_GENERATE_REPORT}' not found")
            self._tracelens_available = False
        
        return self._tracelens_available
    
    def analyze(
        self,
        trace_dir: Path,
        output_dir: Path,
        num_ranks: int = 8,
    ) -> Dict[str, Any]:
        """
        Analyze torch profiler traces using TraceLens CLI.
        
        Runs the following commands based on configuration:
        - TraceLens_generate_perf_report_pytorch (if perf_report_enabled)
        - TraceLens_generate_multi_rank_collective_report_pytorch (if multi_rank_report_enabled)
        
        Args:
            trace_dir: Directory containing torch trace files (*.json.gz)
            output_dir: Output directory for analysis results
            num_ranks: Number of GPU ranks (for multi-rank collective analysis)
        
        Returns:
            Dictionary with analysis results and output paths
        """
        if not self.config.enabled:
            logger.debug("TraceLens analysis is disabled")
            return {"enabled": False}
        
        if not self.is_available():
            logger.warning("TraceLens is not available, skipping analysis")
            return {"enabled": True, "error": "TraceLens not installed"}
        
        results = {
            "enabled": True,
            "trace_dir": str(trace_dir),
            "num_ranks": num_ranks,
            "export_format": self.config.export_format,
            "output_files": [],
            "errors": [],
        }
        
        # Find trace files
        trace_files = self._find_trace_files(trace_dir)
        if not trace_files:
            results["errors"].append(f"No trace files found in {trace_dir}")
            logger.warning(f"No trace files found in {trace_dir}")
            return results
        
        logger.info(f"Found {len(trace_files)} trace files in {trace_dir}")
        
        # Determine output paths based on export format
        use_csv = self.config.export_csv
        use_excel = self.config.export_excel
        
        # 1. Single rank performance report (TraceLens_generate_perf_report_pytorch)
        if self.config.perf_report_enabled:
            logger.info("Running TraceLens_generate_perf_report_pytorch...")
            
            rank0_csv_dir = output_dir / "tracelens_rank0_csvs" if use_csv else None
            rank0_xlsx = output_dir / "tracelens_rank0_report.xlsx" if use_excel else None
            
            if rank0_csv_dir:
                rank0_csv_dir.mkdir(parents=True, exist_ok=True)
            
            trace_file = trace_files[0]
            rank0_result = self._run_generate_report(
                trace_file=trace_file,
                output_csv_dir=rank0_csv_dir,
                output_xlsx=rank0_xlsx,
            )
            results["output_files"].extend(rank0_result.get("files", []))
            if rank0_result.get("error"):
                results["errors"].append(rank0_result["error"])
        
        # 2. Multi-rank collective analysis (TraceLens_generate_multi_rank_collective_report_pytorch)
        if self.config.multi_rank_report_enabled and len(trace_files) >= num_ranks and num_ranks > 1:
            logger.info("Running TraceLens_generate_multi_rank_collective_report_pytorch...")
            
            collective_csv_dir = output_dir / "tracelens_collective_csvs" if use_csv else None
            collective_xlsx = output_dir / "tracelens_collective_report.xlsx" if use_excel else None
            
            if collective_csv_dir:
                collective_csv_dir.mkdir(parents=True, exist_ok=True)
            
            collective_result = self._run_multi_rank_collective(
                trace_dir=trace_dir,
                output_csv_dir=collective_csv_dir,
                output_xlsx=collective_xlsx,
                num_ranks=num_ranks,
            )
            results["output_files"].extend(collective_result.get("files", []))
            if collective_result.get("error"):
                results["errors"].append(collective_result["error"])
        elif self.config.multi_rank_report_enabled and num_ranks > 1:
            logger.info(
                f"Skipping multi-rank analysis: found {len(trace_files)} traces "
                f"but need at least {num_ranks} for world_size={num_ranks}"
            )
        
        logger.info(f"TraceLens analysis complete. Output files: {len(results['output_files'])}")
        return results
    
    def compare_reports(
        self,
        report_dirs: List[Path],
        output_dir: Path,
        labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Compare multiple TraceLens performance reports.
        
        Uses TraceLens_compare_perf_reports_pytorch CLI command.
        
        Args:
            report_dirs: List of directories containing TraceLens CSV reports
            output_dir: Output directory for comparison results
            labels: Optional labels for each report
        
        Returns:
            Dictionary with comparison results
        """
        if len(report_dirs) < 2:
            return {"error": "At least 2 report directories required for comparison"}
        
        if not self.is_available():
            return {"error": "TraceLens not installed"}
        
        result = {"files": [], "error": None}
        
        # Build command (CLI uses underscores, not hyphens)
        cmd = [CLI_COMPARE_REPORTS]
        
        # Add input directories
        for i, report_dir in enumerate(report_dirs):
            cmd.extend(["--input_csvs_dir", str(report_dir)])
            if labels and i < len(labels):
                cmd.extend(["--label", labels[i]])
        
        # Output
        output_csv_dir = output_dir / "tracelens_comparison_csvs"
        output_csv_dir.mkdir(parents=True, exist_ok=True)
        cmd.extend(["--output_csvs_dir", str(output_csv_dir)])
        
        if self.config.export_excel:
            cmd.extend(["--output_xlsx_path", str(output_dir / "tracelens_comparison.xlsx")])
        
        logger.info(f"Running TraceLens compare: {' '.join(cmd)}")
        
        try:
            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,
            )
            
            if proc_result.returncode == 0:
                # Collect output files
                for csv_file in output_csv_dir.glob("*.csv"):
                    result["files"].append(str(csv_file))
                if self.config.export_excel:
                    xlsx_file = output_dir / "tracelens_comparison.xlsx"
                    if xlsx_file.exists():
                        result["files"].append(str(xlsx_file))
                logger.info(f"Comparison complete: {len(result['files'])} files generated")
            else:
                result["error"] = f"Compare failed: {proc_result.stderr}"
                logger.error(f"TraceLens compare failed: {proc_result.stderr}")
                
        except subprocess.TimeoutExpired:
            result["error"] = "Comparison timed out"
            logger.error("TraceLens comparison timed out")
        except Exception as e:
            result["error"] = f"Comparison error: {str(e)}"
            logger.exception(f"TraceLens comparison error: {e}")
        
        return result
    
    def _find_trace_files(self, trace_dir: Path) -> List[Path]:
        """Find all trace files in directory."""
        trace_files = []
        
        # Look for .json.gz and .json files
        for pattern in ["*.json.gz", "*.json"]:
            trace_files.extend(trace_dir.glob(pattern))
        
        # Filter out async_llm traces (main process, not worker)
        worker_traces = [
            f for f in trace_files
            if "async_llm" not in f.name.lower()
        ]
        
        # If we have worker traces, use those; otherwise use all traces
        if worker_traces:
            trace_files = worker_traces
        
        return sorted(trace_files)
    
    def _run_generate_report(
        self,
        trace_file: Path,
        output_csv_dir: Optional[Path] = None,
        output_xlsx: Optional[Path] = None,
    ) -> Dict[str, Any]:
        """
        Run TraceLens_generate_perf_report_pytorch CLI command.
        
        Args:
            trace_file: Path to trace file
            output_csv_dir: Output directory for CSV files (optional)
            output_xlsx: Path for Excel output (optional)
        
        Returns:
            Dictionary with output files and any errors
        """
        result = {"files": [], "error": None}
        
        if not output_csv_dir and not output_xlsx:
            result["error"] = "At least one of output_csv_dir or output_xlsx must be specified"
            return result
        
        # Build command (CLI uses underscores, not hyphens)
        cmd = [
            CLI_GENERATE_REPORT,
            "--profile_json_path", str(trace_file),
        ]
        
        if output_csv_dir:
            cmd.extend(["--output_csvs_dir", str(output_csv_dir)])
        
        if output_xlsx:
            cmd.extend(["--output_xlsx_path", str(output_xlsx)])
        
        # Collective analysis is enabled by default, use --disable_coll_analysis to disable
        if not self.config.collective_analysis:
            cmd.append("--disable_coll_analysis")
        
        if self.config.short_kernel_study:
            cmd.append("--short_kernel_study")
        
        # Enable kernel summary
        cmd.append("--enable_kernel_summary")
        
        if self.config.gpu_arch_config:
            cmd.extend(["--gpu_arch_json_path", self.config.gpu_arch_config])
        
        logger.info(f"Running TraceLens: {' '.join(cmd)}")
        
        try:
            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=600,  # 10 minute timeout
            )
            
            if proc_result.returncode == 0:
                # Collect output files
                if output_csv_dir:
                    for csv_file in output_csv_dir.glob("*.csv"):
                        result["files"].append(str(csv_file))
                if output_xlsx and output_xlsx.exists():
                    result["files"].append(str(output_xlsx))
                logger.info(f"TraceLens perf report complete: {len(result['files'])} files generated")
                if proc_result.stdout:
                    logger.debug(f"TraceLens output: {proc_result.stdout}")
            else:
                result["error"] = f"TraceLens CLI failed: {proc_result.stderr}"
                logger.error(f"TraceLens CLI failed: {proc_result.stderr}")
                if proc_result.stdout:
                    logger.error(f"TraceLens stdout: {proc_result.stdout}")
                    
        except subprocess.TimeoutExpired:
            result["error"] = "TraceLens analysis timed out"
            logger.error("TraceLens analysis timed out")
        except FileNotFoundError:
            result["error"] = f"TraceLens CLI command '{CLI_GENERATE_REPORT}' not found"
            logger.error(f"TraceLens CLI command '{CLI_GENERATE_REPORT}' not found")
        except Exception as e:
            result["error"] = f"TraceLens error: {str(e)}"
            logger.exception(f"TraceLens analysis error: {e}")
        
        return result
    
    def _run_multi_rank_collective(
        self,
        trace_dir: Path,
        output_csv_dir: Optional[Path] = None,
        output_xlsx: Optional[Path] = None,
        num_ranks: int = 8,
    ) -> Dict[str, Any]:
        """
        Run TraceLens_generate_multi_rank_collective_report_pytorch CLI command.
        
        Args:
            trace_dir: Directory containing trace files
            output_csv_dir: Output directory for CSV files
            output_xlsx: Optional path for Excel output
            num_ranks: Number of GPU ranks (world_size)
        
        Returns:
            Dictionary with output files and any errors
        """
        result = {"files": [], "error": None}
        
        if not output_csv_dir and not output_xlsx:
            result["error"] = "At least one of output_csv_dir or output_xlsx must be specified"
            return result
        
        # Find trace pattern - TraceLens expects pattern like "rank*_trace.json.gz"
        # or we can use a pattern that matches our trace files
        trace_files = self._find_trace_files(trace_dir)
        
        if len(trace_files) < num_ranks:
            logger.warning(
                f"Found {len(trace_files)} trace files but num_ranks={num_ranks}. "
                f"Adjusting num_ranks to {len(trace_files)}"
            )
            num_ranks = len(trace_files)
        
        if num_ranks < 2:
            result["error"] = "Multi-rank analysis requires at least 2 ranks"
            return result
        
        # Determine trace pattern
        trace_pattern = self._detect_trace_pattern(trace_dir, trace_files)
        
        # Build command (CLI uses underscores, not hyphens)
        cmd = [CLI_MULTI_RANK_COLLECTIVE]
        
        # Use either --trace_dir or --trace_pattern
        if trace_pattern:
            cmd.extend(["--trace_pattern", trace_pattern])
        else:
            # Fall back to trace_dir if pattern detection fails
            cmd.extend(["--trace_dir", str(trace_dir)])
        
        cmd.extend(["--world_size", str(num_ranks)])
        
        if output_csv_dir:
            cmd.extend(["--output_csvs_dir", str(output_csv_dir)])
        
        if output_xlsx:
            cmd.extend(["--output_xlsx_path", str(output_xlsx)])
        
        logger.info(f"Running TraceLens multi-rank: {' '.join(cmd)}")
        
        try:
            proc_result = subprocess.run(
                cmd,
                capture_output=True,
                text=True,
                timeout=900,  # 15 minute timeout for multi-rank
            )
            
            if proc_result.returncode == 0:
                # Collect output files
                if output_csv_dir:
                    for csv_file in output_csv_dir.glob("*.csv"):
                        result["files"].append(str(csv_file))
                if output_xlsx and output_xlsx.exists():
                    result["files"].append(str(output_xlsx))
                logger.info(f"Multi-rank collective report complete: {len(result['files'])} files generated")
                if proc_result.stdout:
                    logger.debug(f"TraceLens output: {proc_result.stdout}")
            else:
                result["error"] = f"Multi-rank analysis failed: {proc_result.stderr}"
                logger.error(f"TraceLens multi-rank failed: {proc_result.stderr}")
                if proc_result.stdout:
                    logger.error(f"TraceLens stdout: {proc_result.stdout}")
                    
        except subprocess.TimeoutExpired:
            result["error"] = "Multi-rank analysis timed out"
            logger.error("TraceLens multi-rank analysis timed out")
        except FileNotFoundError:
            result["error"] = f"TraceLens CLI command '{CLI_MULTI_RANK_COLLECTIVE}' not found"
            logger.error(f"TraceLens CLI command '{CLI_MULTI_RANK_COLLECTIVE}' not found")
        except Exception as e:
            result["error"] = f"Multi-rank analysis error: {str(e)}"
            logger.exception(f"TraceLens multi-rank error: {e}")
        
        return result
    
    def _detect_trace_pattern(
        self,
        trace_dir: Path,
        trace_files: List[Path],
    ) -> Optional[str]:
        """
        Detect trace file pattern for multi-rank analysis.
        
        TraceLens expects a pattern like:
        - "/path/to/traces/rank*_trace.json.gz"
        - "/path/to/traces/worker*_trace.json"
        
        Args:
            trace_dir: Directory containing traces
            trace_files: List of found trace files
        
        Returns:
            Pattern string or None if no pattern detected
        """
        if not trace_files:
            return None
        
        # Try to detect common patterns
        first_name = trace_files[0].name
        
        import re
        
        # Pattern 1: rank-{N} or rank{N} (e.g., rank-0, rank0)
        rank_match = re.search(r'rank[-_]?(\d+)', first_name, re.IGNORECASE)
        if rank_match:
            # Replace rank number with wildcard, preserving separator
            pattern_name = re.sub(r'rank([-_]?)\d+', r'rank\1*', first_name, flags=re.IGNORECASE)
            return str(trace_dir / pattern_name)
        
        # Pattern 2: worker-{N} or worker{N}
        worker_match = re.search(r'worker[-_]?(\d+)', first_name, re.IGNORECASE)
        if worker_match:
            pattern_name = re.sub(r'worker([-_]?)\d+', r'worker\1*', first_name, flags=re.IGNORECASE)
            return str(trace_dir / pattern_name)
        
        # Pattern 3: gpu-{N} or gpu{N}
        gpu_match = re.search(r'gpu[-_]?(\d+)', first_name, re.IGNORECASE)
        if gpu_match:
            pattern_name = re.sub(r'gpu([-_]?)\d+', r'gpu\1*', first_name, flags=re.IGNORECASE)
            return str(trace_dir / pattern_name)
        
        # Pattern 4: Check for sequential numbering in filename
        # e.g., trace_0.json, trace_1.json
        num_match = re.search(r'[._-](\d+)[._-]', first_name)
        if num_match:
            # Check if we have sequential files
            num_str = num_match.group(1)
            pattern_name = first_name.replace(num_str, '*', 1)
            return str(trace_dir / pattern_name)
        
        # No pattern detected
        logger.warning(f"Could not detect trace pattern from: {first_name}")
        return None


def run_tracelens_analysis(
    config: TraceLensConfig,
    trace_dir: Path,
    output_dir: Path,
    num_ranks: int = 8,
) -> Dict[str, Any]:
    """
    Convenience function to run TraceLens analysis.
    
    Args:
        config: TraceLens configuration
        trace_dir: Directory containing torch trace files
        output_dir: Output directory for analysis results
        num_ranks: Number of GPU ranks
    
    Returns:
        Analysis results dictionary
    """
    analyzer = TraceLensAnalyzer(config)
    return analyzer.analyze(trace_dir, output_dir, num_ranks)


def compare_tracelens_reports(
    config: TraceLensConfig,
    report_dirs: List[Path],
    output_dir: Path,
    labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Convenience function to compare TraceLens reports.
    
    Args:
        config: TraceLens configuration
        report_dirs: List of directories containing TraceLens CSV reports
        output_dir: Output directory for comparison results
        labels: Optional labels for each report
    
    Returns:
        Comparison results dictionary
    """
    analyzer = TraceLensAnalyzer(config)
    return analyzer.compare_reports(report_dirs, output_dir, labels)
