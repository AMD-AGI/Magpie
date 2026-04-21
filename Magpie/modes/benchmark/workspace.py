###############################################################################
# Copyright (c) 2026 Advanced Micro Devices, Inc. All rights reserved.
#
# See LICENSE for license information.
###############################################################################
"""
Workspace manager for benchmark mode.

Manages shared directories for benchmark execution and result collection.
"""

import json
import logging
import shutil
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

import yaml

logger = logging.getLogger(__name__)


class WorkspaceManager:
    """
    Manages workspace directories for benchmark tasks.
    
    Creates structured directories for each benchmark run:
    - config.yaml: Configuration snapshot
    - torch_trace/: PyTorch profiler output
    - system_profile/: System profiler output (rocprof/ncu)
    - inferencex_result.json: InferenceX raw output
    - server.log: Server logs
    - benchmark_report.json: Magpie summary report
    """
    
    def __init__(
        self,
        base_dir: str = "./results",
        framework: str = "benchmark",
    ):
        """
        Initialize workspace manager.
        
        Args:
            base_dir: Base directory for all benchmark results
            framework: Framework name for directory naming
        """
        self.base_dir = Path(base_dir)
        self.framework = framework
        self._workspace_path: Optional[Path] = None
    
    def create(self, config: Optional[Dict[str, Any]] = None) -> Path:
        """
        Create a new workspace directory for a benchmark run.
        
        Args:
            config: Optional configuration to save as snapshot
        
        Returns:
            Path to the created workspace directory (absolute path)
        """
        # Generate timestamp-based name
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        workspace_name = f"benchmark_{self.framework}_{timestamp}"
        
        # Create workspace directory with absolute path (required for Docker volume mounts)
        workspace_path = (self.base_dir / workspace_name).resolve()
        workspace_path.mkdir(parents=True, exist_ok=True)
        
        # Create subdirectories
        (workspace_path / "torch_trace").mkdir(exist_ok=True)
        (workspace_path / "system_profile").mkdir(exist_ok=True)
        
        # Save configuration snapshot
        if config:
            self._save_config_snapshot(workspace_path, config)
        
        self._workspace_path = workspace_path
        logger.info(f"Created benchmark workspace: {workspace_path}")
        
        return workspace_path
    
    def _save_config_snapshot(self, workspace_path: Path, config: Dict[str, Any]) -> None:
        """Save configuration snapshot to workspace."""
        config_file = workspace_path / "config.yaml"
        try:
            with open(config_file, 'w') as f:
                yaml.dump(config, f, default_flow_style=False)
            logger.debug(f"Saved config snapshot: {config_file}")
        except Exception as e:
            logger.warning(f"Failed to save config snapshot: {e}")
    
    @property
    def workspace_path(self) -> Optional[Path]:
        """Get current workspace path."""
        return self._workspace_path
    
    @property
    def torch_trace_dir(self) -> Optional[Path]:
        """Get torch_trace directory path."""
        if self._workspace_path:
            return self._workspace_path / "torch_trace"
        return None
    
    @property
    def system_profile_dir(self) -> Optional[Path]:
        """Get system_profile directory path."""
        if self._workspace_path:
            return self._workspace_path / "system_profile"
        return None
    
    def get_result_file_path(self, filename: str = "inferencex_result.json") -> Optional[Path]:
        """Get path for a result file in workspace."""
        if self._workspace_path:
            return self._workspace_path / filename
        return None
    
    def save_report(self, report: Dict[str, Any], filename: str = "benchmark_report.json") -> None:
        """
        Save benchmark report to workspace.
        
        Args:
            report: Report data to save
            filename: Output filename
        """
        if not self._workspace_path:
            logger.error("No workspace created, cannot save report")
            return
        
        report_file = self._workspace_path / filename
        try:
            with open(report_file, 'w') as f:
                json.dump(report, f, indent=2)
            logger.info(f"Saved benchmark report: {report_file}")
        except Exception as e:
            logger.error(f"Failed to save report: {e}")
    
    def save_summary(self, summary: str, filename: str = "summary.txt") -> None:
        """
        Save human-readable summary to workspace.
        
        Args:
            summary: Summary text
            filename: Output filename
        """
        if not self._workspace_path:
            logger.error("No workspace created, cannot save summary")
            return
        
        summary_file = self._workspace_path / filename
        try:
            with open(summary_file, 'w') as f:
                f.write(summary)
            logger.info(f"Saved summary: {summary_file}")
        except Exception as e:
            logger.error(f"Failed to save summary: {e}")
    
    def collect_results(self) -> Dict[str, Any]:
        """
        Collect all results from workspace.
        
        Returns:
            Dictionary containing all collected results
        """
        if not self._workspace_path:
            return {}
        
        results = {
            "workspace_path": str(self._workspace_path),
            "inferencex_result": None,
            "torch_trace_files": [],
            "system_profile_files": [],
            "server_log": None,
        }
        
        # Read benchmark result
        result_file = self._workspace_path / "inferencex_result.json"
        if result_file.exists():
            try:
                with open(result_file, 'r') as f:
                    results["inferencex_result"] = json.load(f)
            except Exception as e:
                logger.warning(f"Failed to read benchmark result: {e}")
        
        # List torch trace files
        torch_trace_dir = self._workspace_path / "torch_trace"
        if torch_trace_dir.exists():
            results["torch_trace_files"] = [
                str(f) for f in torch_trace_dir.iterdir() if f.is_file()
            ]
        
        # List system profile files
        system_profile_dir = self._workspace_path / "system_profile"
        if system_profile_dir.exists():
            results["system_profile_files"] = [
                str(f) for f in system_profile_dir.iterdir() if f.is_file()
            ]
        
        # Read server log
        server_log = self._workspace_path / "server.log"
        if server_log.exists():
            try:
                with open(server_log, 'r') as f:
                    results["server_log"] = f.read()
            except Exception as e:
                logger.warning(f"Failed to read server log: {e}")
        
        return results
    
    def cleanup(self, keep_results: bool = True) -> None:
        """
        Clean up workspace.
        
        Args:
            keep_results: If True, keep result files but remove temp files
                         If False, remove entire workspace
        """
        if not self._workspace_path:
            return
        
        if keep_results:
            # Remove only temporary files
            temp_patterns = ["*.tmp", "*.pid", "*.lock"]
            for pattern in temp_patterns:
                for f in self._workspace_path.glob(pattern):
                    try:
                        f.unlink()
                    except Exception:
                        pass
        else:
            # Remove entire workspace
            try:
                shutil.rmtree(self._workspace_path)
                logger.info(f"Removed workspace: {self._workspace_path}")
            except Exception as e:
                logger.error(f"Failed to remove workspace: {e}")
        
        self._workspace_path = None
    
    @staticmethod
    def list_workspaces(base_dir: str = "./results") -> list:
        """
        List all benchmark workspaces in base directory.
        
        Args:
            base_dir: Base directory to search
        
        Returns:
            List of workspace directory paths
        """
        base_path = Path(base_dir)
        if not base_path.exists():
            return []
        
        workspaces = []
        for d in base_path.iterdir():
            if d.is_dir() and d.name.startswith("benchmark_"):
                workspaces.append(str(d))
        
        return sorted(workspaces, reverse=True)  # Most recent first
