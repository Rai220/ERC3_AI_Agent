from dotenv import load_dotenv, find_dotenv
load_dotenv(find_dotenv())

import textwrap
import argparse
from openai import OpenAI
from store_agent import run_agent
from erc3 import ERC3

# Parse command line arguments
parser = argparse.ArgumentParser(description='Run ERC3 Agent tests')
parser.add_argument('--only', type=int, metavar='X', 
                    help='Run only test number X (1-based indexing)')
parser.add_argument('--fail-fast', action='store_true',
                    help='Stop on first failed test')
args = parser.parse_args()

client = OpenAI()
core = ERC3()
# MODEL_ID = "gpt-5.1"
MODEL_ID = "gpt-5.2"

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

for idx, task in enumerate(tasks_to_run, start=(args.only if args.only else 1)):
    print("="*40)
    print(f"Starting Task #{idx}: {task.task_id} ({task.spec_id}): {task.task_text}")
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
        if result.eval.score > 0:
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
            # Останавливаемся при первом провале если указан --fail-fast
            if args.fail_fast:
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

# Отправляем сессию если был полный прогон (без --only и без преждевременной остановки)
if args.only is not None:
    print(f"Skipping session submission (only test #{args.only} was run)")
elif args.fail_fast and failed_tests > 0:
    print(f"Skipping session submission (stopped early due to --fail-fast)")
else:
    # Полный прогон - подаём независимо от результатов
    core.submit_session(res.session_id)
    print("Session submitted successfully!")
