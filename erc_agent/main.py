from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import textwrap
import argparse
import threading
from concurrent.futures import ThreadPoolExecutor, as_completed
from openai import OpenAI
from agent import run_agent
from erc3 import ERC3
from langchain_gigachat import GigaChat
from api_logger import init_run_logger, finalize_task, get_run_logger, register_task

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run ERC3 Agent tests')
parser.add_argument('--only', type=int, metavar='X', 
                    help='Run only test number X (1-based indexing)')
parser.add_argument('--fail-fast', action='store_true',
                    help='Stop on first failed test')
parser.add_argument('--concurrency', type=int, default=1, metavar='N',
                    help='Number of tasks to run in parallel (default: 1)')
args = parser.parse_args()

core = ERC3()
MODEL_ID = "gpt-5.2"
# MODEL_ID = "GigaChat-3-Ultra"

if MODEL_ID.startswith("GigaChat"):
    client = GigaChat(model=MODEL_ID, timeout=600)
else:
    client = OpenAI()

# Start session with metadata
# Флаги: compete_accuracy - для призового соревнования 9 декабря
# Другие флаги: compete_budget, compete_speed, compete_local (отдельные leaderboards)
res = core.start_session(
    # benchmark="erc3-prod",
    benchmark="erc3-prod",
    workspace="my",
    name=f"@Krestnikov (Giga team)",
    architecture="React + think-tool + Structured reasoning",
    flags=["compete_accuracy"]
)

status = core.session_status(res.session_id)
print(f"Session has {len(status.tasks)} tasks")

# Инициализируем JSONL логгер для этого прогона
run_logger = init_run_logger(MODEL_ID)
run_logger.log_session_info(res.session_id, "erc3-prod", len(status.tasks))
print(f"Logs will be saved to: {run_logger.get_run_dir()}")

# Handle --only option
if args.only is not None:
    if args.only < 1 or args.only > len(status.tasks):
        print(f"Error: Test number {args.only} is out of range (1-{len(status.tasks)})")
        exit(1)
    print(f"Running only test #{args.only}")
    tasks_to_run = [status.tasks[args.only - 1]]
else:
    tasks_to_run = status.tasks

# Счетчики для статистики тестов
passed_tests = 0
failed_tests = 0
failed_task_details = []  # Список для хранения информации о проваленных тестах
stats_lock = threading.Lock()  # Для потокобезопасного обновления статистики
stop_flag = threading.Event()  # Флаг для остановки при --fail-fast


def run_single_task(idx: int, task) -> dict:
    """Выполняет одну задачу и возвращает результат"""
    if stop_flag.is_set():
        return None
    
    print("="*40)
    print(f"Starting Task #{idx}: {task.task_id} ({task.spec_id}): {task.task_text}")
    
    # Регистрируем задачу в логгере с порядковым номером
    register_task(task.task_id, idx)
    
    # start the task
    core.start_task(task)
    try:
        run_agent(MODEL_ID, core, task)
    except Exception as e:
        print(e)
    result = core.complete_task(task)
    
    task_result = {
        'idx': idx,
        'task': task,
        'result': result,
    }
    
    if result.eval:
        explain = textwrap.indent(result.eval.logs, "  ")
        print(f"\nSCORE: {result.eval.score}\n{explain}\n")
        task_result['passed'] = result.eval.score > 0
        task_result['score'] = result.eval.score
        task_result['logs'] = result.eval.logs
    else:
        task_result['passed'] = False
        task_result['score'] = 0
        task_result['logs'] = "No evaluation result"
    
    # Финализируем файл лога задачи (переименовываем в GOOD/BAD)
    finalize_task(task.task_id, task_result['passed'])
    
    return task_result


if args.concurrency > 1:
    # Параллельное выполнение
    print(f"Running tasks with concurrency={args.concurrency}")
    
    # Создаём список (idx, task) для запуска
    indexed_tasks = list(enumerate(tasks_to_run, start=(args.only if args.only else 1)))
    
    with ThreadPoolExecutor(max_workers=args.concurrency) as executor:
        # Отправляем все задачи на выполнение
        futures = {executor.submit(run_single_task, idx, task): (idx, task) 
                   for idx, task in indexed_tasks}
        
        for future in as_completed(futures):
            task_result = future.result()
            
            if task_result is None:
                continue
            
            with stats_lock:
                if task_result['passed']:
                    passed_tests += 1
                else:
                    failed_tests += 1
                    failed_task_details.append({
                        'idx': task_result['idx'],
                        'spec_id': task_result['task'].spec_id,
                        'task_text': task_result['task'].task_text[:60] + '...' 
                                     if len(task_result['task'].task_text) > 60 
                                     else task_result['task'].task_text,
                        'reason': task_result['logs'].strip()
                    })
                    
                    if args.fail_fast:
                        print(f"\n🛑 ОСТАНОВКА: Тест #{task_result['idx']} провален (--fail-fast)")
                        stop_flag.set()
                        # Отменяем оставшиеся задачи
                        for f in futures:
                            f.cancel()
                        break
else:
    # Последовательное выполнение (оригинальное поведение)
    for idx, task in enumerate(tasks_to_run, start=(args.only if args.only else 1)):
        print("="*40)
        print(f"Starting Task #{idx}: {task.task_id} ({task.spec_id}): {task.task_text}")
        # Регистрируем задачу в логгере с порядковым номером
        register_task(task.task_id, idx)
        # start the task
        core.start_task(task)
        try:
            run_agent(MODEL_ID, core, task)
        except Exception as e:
            print(e)
        result = core.complete_task(task)
        if result.eval:
            explain = textwrap.indent(result.eval.logs, "  ")
            print(f"\nSCORE: {result.eval.score}\n{explain}\n")
            
            # Подсчитываем пройденные/непройденные тесты
            task_passed = result.eval.score > 0
            if task_passed:
                passed_tests += 1
            else:
                failed_tests += 1
                # Сохраняем информацию о проваленном тесте
                failed_task_details.append({
                    'idx': idx,
                    'spec_id': task.spec_id,
                    'task_text': task.task_text[:60] + '...' if len(task.task_text) > 60 else task.task_text,
                    'reason': result.eval.logs.strip()
                })
            
            # Финализируем файл лога задачи (переименовываем в GOOD/BAD)
            finalize_task(task.task_id, task_passed)
            
            # Останавливаемся при первом провале если указан --fail-fast
            if not task_passed and args.fail_fast:
                print(f"\n🛑 ОСТАНОВКА: Тест #{idx} провален (--fail-fast)")
                break

# Выводим статистику тестов
print("="*40)
print(f"РЕЗУЛЬТАТЫ ТЕСТИРОВАНИЯ:")
print(f"  Пройдено: {passed_tests}")
print(f"  Не пройдено: {failed_tests}")
print(f"  Всего: {passed_tests + failed_tests}")
print("="*40)

# Выводим список проваленных тестов
if failed_task_details:
    print("\n❌ СПИСОК ПРОВАЛЕННЫХ ТЕСТОВ:")
    print("-"*40)
    for fail in failed_task_details:
        print(f"  #{fail['idx']} ({fail['spec_id']})")
        print(f"     Задача: {fail['task_text']}")
        print(f"     Причина: {fail['reason']}")
        print()
    print("-"*40)

# Сортируем проваленные тесты по номеру для удобства
failed_task_details.sort(key=lambda x: x['idx'])

# Сохраняем итоговые результаты в JSONL директорию
if run_logger:
    run_logger.log_session_results(passed_tests, failed_tests, failed_task_details)
    print(f"\n📁 Логи сохранены в: {run_logger.get_run_dir()}")

# Отправляем сессию если был полный прогон (без --only и без преждевременной остановки)
if args.only is not None:
    print(f"Skipping session submission (only test #{args.only} was run)")
elif args.fail_fast and failed_tests > 0:
    print(f"Skipping session submission (stopped early due to --fail-fast)")
else:
    # Полный прогон - подаём независимо от результатов
    core.submit_session(res.session_id)
    print("Session submitted successfully!")
