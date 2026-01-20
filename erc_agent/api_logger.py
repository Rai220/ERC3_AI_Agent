"""
JSONL логгер для API запросов/ответов.

Создаёт структуру директорий:
  logs/
    {YYYY-MM-DD_HH-MM-SS}_{model_name}/
      01_task.jsonl                    - во время выполнения
      01_GOOD_2025-12-17_18-30-45.jsonl - если задача решена
      01_BAD_2025-12-17_18-30-45.jsonl  - если задача не решена
"""

import json
import os
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional, Tuple
import threading


class RunLogger:
    """Логгер для одного прогона (сессии)."""
    
    def __init__(self, model_name: str, base_dir: str = "logs"):
        """
        Инициализирует логгер для прогона.
        
        Args:
            model_name: Название модели (будет в имени папки)
            base_dir: Базовая директория для логов
        """
        self.model_name = self._sanitize_name(model_name)
        self.base_dir = Path(base_dir)
        
        # Сохраняем timestamp прогона для использования в именах файлов
        self.run_timestamp = datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
        self.run_dir = self.base_dir / f"{self.run_timestamp}_{self.model_name}"
        
        # Создаём директорию
        self.run_dir.mkdir(parents=True, exist_ok=True)
        
        # Хранилище: task_id -> (file_path, task_number)
        self._task_files: Dict[str, Tuple[Path, int]] = {}
        self._lock = threading.Lock()
        
    def _sanitize_name(self, name: str) -> str:
        """Очищает имя для использования в названии файла/папки."""
        # Заменяем проблемные символы
        sanitized = name.replace("/", "_").replace("\\", "_").replace(":", "_")
        sanitized = sanitized.replace(" ", "_").replace(".", "-")
        return sanitized
    
    def get_run_dir(self) -> Path:
        """Возвращает путь к директории прогона."""
        return self.run_dir
    
    def register_task(self, task_id: str, task_number: int):
        """
        Регистрирует задачу с её порядковым номером.
        
        Args:
            task_id: ID задачи
            task_number: Порядковый номер задачи (1, 2, 3...)
        """
        with self._lock:
            # Формат: {номер}_task.jsonl (во время выполнения)
            file_name = f"{task_number:02d}_task.jsonl"
            file_path = self.run_dir / file_name
            self._task_files[task_id] = (file_path, task_number)
    
    def log_api_call(
        self,
        task_id: str,
        call_type: str,  # "request" или "response" или "llm_request" или "llm_response"
        tool_name: str,
        data: Any,
        timestamp: Optional[datetime] = None
    ):
        """
        Записывает API вызов в JSONL файл задачи.
        
        Args:
            task_id: ID задачи
            call_type: Тип записи (request, response, llm_request, llm_response, error)
            tool_name: Название инструмента/API метода
            data: Данные для записи (будут сериализованы в JSON)
            timestamp: Временная метка (по умолчанию - текущее время)
        """
        if timestamp is None:
            timestamp = datetime.now()
        
        # Формируем запись
        entry = {
            "timestamp": timestamp.isoformat(),
            "type": call_type,
            "tool": tool_name,
            "data": data
        }
        
        with self._lock:
            # Получаем путь к файлу задачи
            if task_id not in self._task_files:
                # Если задача не была зарегистрирована, создаём временный файл
                file_path = self.run_dir / f"unknown_{task_id}.jsonl"
                self._task_files[task_id] = (file_path, 0)
            
            file_path, _ = self._task_files[task_id]
            
            # Записываем в файл
            with open(file_path, "a", encoding="utf-8") as f:
                json.dump(entry, f, ensure_ascii=False, default=str)
                f.write("\n")
    
    def finalize_task(self, task_id: str, success: bool):
        """
        Финализирует файл задачи, переименовывая его с результатом.
        
        Формат: {номер}_{GOOD/BAD}_{дата-время_прогона}.jsonl
        
        Args:
            task_id: ID задачи
            success: True если задача решена, False если нет
        """
        with self._lock:
            if task_id not in self._task_files:
                return
            
            old_path, task_number = self._task_files[task_id]
            if not old_path.exists():
                return
            
            # Формируем новое имя: {номер}_{GOOD/BAD}_{дата-время}.jsonl
            result = "GOOD" if success else "BAD"
            new_name = f"{task_number:02d}_{result}_{self.run_timestamp}.jsonl"
            new_path = self.run_dir / new_name
            
            # Переименовываем файл
            old_path.rename(new_path)
            self._task_files[task_id] = (new_path, task_number)
    
    def log_session_info(self, session_id: str, benchmark: str, tasks_count: int):
        """
        Записывает информацию о сессии в отдельный файл.
        
        Args:
            session_id: ID сессии
            benchmark: Название бенчмарка
            tasks_count: Количество задач
        """
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
        """
        Записывает итоговые результаты сессии.
        
        Args:
            passed: Количество пройденных тестов
            failed: Количество проваленных тестов
            failed_details: Детали проваленных тестов
        """
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


# Глобальный экземпляр логгера для текущего прогона
_current_run_logger: Optional[RunLogger] = None
_logger_lock = threading.Lock()


def init_run_logger(model_name: str, base_dir: str = "logs") -> RunLogger:
    """
    Инициализирует глобальный логгер для прогона.
    
    Args:
        model_name: Название модели
        base_dir: Базовая директория для логов
    
    Returns:
        Инициализированный RunLogger
    """
    global _current_run_logger
    with _logger_lock:
        _current_run_logger = RunLogger(model_name, base_dir)
        return _current_run_logger


def get_run_logger() -> Optional[RunLogger]:
    """Возвращает текущий логгер прогона."""
    return _current_run_logger


def register_task(task_id: str, task_number: int):
    """
    Регистрирует задачу с её порядковым номером.
    Использует глобальный логгер если он инициализирован.
    """
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
    """
    Удобная функция для логирования API вызова.
    Использует глобальный логгер если он инициализирован.
    """
    logger = get_run_logger()
    if logger:
        logger.log_api_call(task_id, call_type, tool_name, data, timestamp)


def finalize_task(task_id: str, success: bool):
    """
    Удобная функция для финализации задачи.
    Использует глобальный логгер если он инициализирован.
    """
    logger = get_run_logger()
    if logger:
        logger.finalize_task(task_id, success)
