import modal
from modal import App, Image, Secret, Volume
from pathlib import Path
import structlog

sabiyarn = Image.debian_slim(python_version="3.12").pip_install(
    "transformers[torch]",
    "bitsandbytes",
    "datasets",
    "wandb",
    "structlog",
    "PyYAML",
    "simple-parsing==0.0.3rc1",
    "sentencepiece",
    "fairscale",
    # force_build=True
)

LOG = structlog.stdlib.get_logger()
VOL_MOUNT_PATH = Path("/vol")
stub = App(name="sabiyarn-ablation-tests", image=sabiyarn)
output_vol = Volume.from_name("sabiyarn_v2", create_if_missing=True)

restart_tracker_dict = modal.Dict.from_name("sabiyarn-ablation", create_if_missing=True)


def track_restarts(restart_tracker: modal.Dict) -> int:
    if not restart_tracker.contains("count"):
        preemption_count = 0
        print(f"Starting first time. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    else:
        preemption_count = restart_tracker.get("count") + 1
        print(f"Restarting after pre-emption. {preemption_count=}")
        restart_tracker["count"] = preemption_count
    return preemption_count


def prepare_train():
    from ..data import prepare

    prepare.run()


def train():
    import train
    LOG.info("starting training runs")
    train.train()


@stub.function(
    gpu=modal.gpu.A10G(count=2),
    timeout=60 * 60 * 4,
    cpu=8.0,
    secrets=[Secret.from_name("wandb-api"), Secret.from_name("hf-secret")],
    volumes={VOL_MOUNT_PATH: output_vol},
)
def run():
    LOG.info("modal instance running..")
    prepare_train()
    train()
