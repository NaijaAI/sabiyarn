import modal
from modal import Stub, Image, Secret, Volume
from pathlib import Path
import structlog

sabiyarn_image = (
    Image.debian_slim(python_version="3.12")
    .pip_install(
        "transformers[torch]",
        'datasets',
        'wandb',
        'structlog',
        'PyYAML'
    )
)

LOG = structlog.stdlib.get_logger()
VOL_MOUNT_PATH = Path("/vol")
stub = Stub(name='sabiyarn-ablation',
            image=sabiyarn_image)
output_vol = Volume.from_name('sabiyarn_v2',
                              create_if_missing=True)

restart_tracker_dict = modal.Dict.from_name(
    "sabiyarn-ablation", create_if_missing=True
)


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


def train():
    import train

    LOG.info("starting ablation test..")
    train.train()


@stub.function(
    gpu="A10g",
    timeout=60*60*4,
    secrets=[Secret.from_name('wandb-api')],
    volumes={VOL_MOUNT_PATH: output_vol},
    _allow_background_volume_commits=True
)
def run():
    LOG.info("modal instance running..")
    prepare_train()
    train()
