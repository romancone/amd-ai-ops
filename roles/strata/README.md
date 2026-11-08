# Strata deployment

The [playbook](../../strata.yml) installs Strata on the `amd_hosts` inventory group. In [group_vars/amd_hosts.yml](../../group_vars/amd_hosts.yml), set `strata_default_model` to the model to serve. Set `strata_models` only to prepare additional models.

Run from the repository root:

```bash
ansible-playbook -i hosts strata.yml --diff
```

| File | Purpose |
|---|---|
| [defaults/main.yml](defaults/main.yml) | Default model, context, concurrency, memory and storage settings |
| [tasks/main.yml](tasks/main.yml) | Install prerequisites, check out Strata, start the selected instance and verify the live API |
| [tasks/model.yml](tasks/model.yml) | Run upstream setup when preparation settings change and validate the generated configuration |
| [tasks/serving.yml](tasks/serving.yml) | Set the IQ3_S conversation cache and output budget while preserving other settings |
| [templates/strata@.service.j2](templates/strata@.service.j2), [handlers/main.yml](handlers/main.yml) | Define the systemd service and restart it when its configuration changes |
| [templates/nvme.rules.j2](templates/nvme.rules.j2) | Set the scheduler for explicitly selected NVMe devices |
| [files/verify_model.py](files/verify_model.py) | Verify restored model shards when `strata_verify_restored_weights` is enabled |
