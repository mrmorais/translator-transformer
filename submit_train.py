from ray.job_submission import JobSubmissionClient
import os

client = JobSubmissionClient("http://127.0.0.1:8265")

entrypoint = (
    "rm -rf translator-transformer; "
    "git clone https://github.com/mrmorais/translator-transformer || true; "
    "python train_distr.py"
)

submission_id = client.submit_job(
    entrypoint=entrypoint,
    runtime_env={
        "pip": ["torch", "ray[train]", "tokenizers", "wandb", "tqdm", "pandas"],
        "env_vars": {
            "WANDB_API_KEY": os.environ["WANDB_API_KEY"],
        }
    },
)

print("Use the following command to follow this Job's logs:")
print(f"ray job logs '{submission_id}' --follow")
