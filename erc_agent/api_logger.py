"""JSONL logger for API requests/responses"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import threading


class RunLogger:
    """Logger for a single run session"""
    
    def __init__(self, model_name: str, base_dir: str = "logs"):
        self.model_name = self._sanitize_name(model_name)
        self.base_dir = Path(base_dir)
        
        # Сохраняем timestamp прогона для использования в именах файлов
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.base_dir / f"{self.run_timestamp}_{self.model_name}"
        
        # Создаём директорию
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        self._task_files: Dict[str, Tuple[Path, int]] = {}
        self._lock = threading.Lock()
        
    def _sanitize_name(self, name: str) -> str:
        sanitized = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        sanitized = sanitized.replace(" ", "_").replace(".", "-")
        return sanitized
    
    def get_run_dir(self) -> Path:
        return self.run_dir
    
    def register_task(self, task_id: str, task_number: int):
        with self._lock:
            file_name = f"{task_number:02d}_task.jsonl"
            file_path = self.run_dir / file_name
            self._task_files[task_id] = (file_path, task_number)
    
    def log_api_call(
        self,
        task_id: str,
        call_type: str,
        tool_name: str,
        data: Any,
        timestamp: Optional[datetime] = None
    ):
        if timestamp is None:
            timestamp = datetime.now()
        
        entry = {
            "timestamp": timestamp.isoformat(),
            "type": call_type,
            "tool": tool_name,
            "data": data
        }
        
        with self._lock:
            if task_id not in self._task_files:
                file_path = self.run_dir / f"unknown_{task_id}.jsonl"
                self._task_files[task_id] = (file_path, 0)
            
            file_path, _ = self._task_files[task_id]
            
            with open(file_path, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
                f.write("\n")
    
    def finalize_task(self, task_id: str, success: bool):
        with self._lock:
            if task_id not in self._task_files:
                return
            
            old_path, task_number = self._task_files[task_id]
            if not old_path.exists():
                return
            
            result = "GOOD" if success else "BAD"
            new_name = f"{task_number:02d}_{result}_{self.run_timestamp}.jsonl"
            new_path = self.run_dir / new_name
            old_path.rename(new_path)
            self._task_files[task_id] = (new_path, task_number)
    
    def log_session_info(self, session_id: str, benchmark: str, tasks_count: int):
        info = {
            "session_id": session_id,
            "benchmark": benchmark,
            "model": self.model_name,
            "tasks_count": tasks_count,
            "started_at": datetime.now().isoformat()
        }
        
        info_path = self.run_dir / "session_info.json"
        with open(info_path, "w", encoding="utf-8") as f:
            json.dump(info, f, ensure_ascii=False, indent=2)
    
    def log_session_results(self, passed: int, failed: int, failed_details: list):
        results = {
            "completed_at": datetime.now().isoformat(),
            "passed": passed,
            "failed": failed,
            "total": passed + failed,
            "success_rate": passed / (passed + failed) if (passed + failed) > 0 else 0,
            "failed_details": failed_details
        }
        
        results_path = self.run_dir / "session_results.json"
        with open(results_path, "w", encoding="utf-8") as f:
            json.dump(results, f, ensure_ascii=False, indent=2)


_current_run_logger: Optional[RunLogger] = None
_logger_lock = threading.Lock()


def init_run_logger(model_name: str, base_dir: str = "logs") -> RunLogger:
    global _current_run_logger
    with _logger_lock:
        _current_run_logger = RunLogger(model_name, base_dir)
        return _current_run_logger


def get_run_logger() -> Optional[RunLogger]:
    return _current_run_logger


def register_task(task_id: str, task_number: int):
    logger = get_run_logger()
    if logger:
        logger.register_task(task_id, task_number)


def log_api_call(
    task_id: str,
    call_type: str,
    tool_name: str,
    data: Any,
    timestamp: Optional[datetime] = None
):
    logger = get_run_logger()
    if logger:
        logger.log_api_call(task_id, call_type, tool_name, data, timestamp)


def finalize_task(task_id: str, success: bool):
    logger = get_run_logger()
    if logger:
        logger.finalize_task(task_id, success)
