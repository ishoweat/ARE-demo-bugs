import time

GPU_BUSY = True


def run_gpu_job(job_name):
    if GPU_BUSY:
        raise RuntimeError(
            f"GPU Busy: cannot start {job_name}"
        )

    print(f"Starting {job_name}")
    time.sleep(2)
    print(f"Finished {job_name}")


if __name__ == "__main__":
    run_gpu_job("Inference Job")
