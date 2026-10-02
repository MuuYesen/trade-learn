"""Lightweight spawn targets that do not import the benchmark or numeric stack."""


def send_result_then_stall(queue):
    import time

    queue.put({"status": "success", "final_value": 100.0})
    time.sleep(60)
