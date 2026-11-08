# Strata local inference

Serve Qwen3.8-Flash-Next on an AMD GPU using HIP. The default is IQ3_S with two 256K slots, q4_0 KV, a shared 2 GiB conversation cache and 16K output per request. Setup and inference leave 10 GiB of system RAM for the OS.

| Model quant | When to use it | Download | Expert arena |
|---|---|---:|---:|
| IQ3_S | Default for coding and long conversations | 83.6 GB | 50.3 GB |
| IQ2_XS | Smallest download and fastest decode of these four | 68.0 GB | 34.0 GB |
| IQ3_XXS | Smaller 3-bit alternative to IQ3_S | 75.8 GB | 42.9 GB |
| UD-IQ4_XS | 4-bit option. This profile reads some experts from SSD | 93.7 GB | 59.5 GB |

Sizes are decimal GB. The expert arena holds model weights and is only part of runtime memory. [Measured results](reports/strata.html)

**Ansible.** Edit [group_vars/amd_hosts.yml](group_vars/amd_hosts.yml):

```yaml
strata_default_model: IQ3_S       # Prepare and serve this model
# strata_models: [IQ3_S, IQ2_XS]  # Also prepare IQ2_XS for switching
# strata_verify_restored_weights: true  # Check shards copied from another installation
```

```bash { name=strata-ansible }
ansible-playbook -i hosts strata.yml --diff
```

## 1. Manual installation

For a fresh IQ3_S installation, run these cells in order from this repository on Ubuntu x86_64 with a working AMD GPU driver. For an existing installation, rerun the playbook above. Installation needs 46,000 MiB available RAM and 350 GiB free on the model filesystem.

### 1.1 Set up IQ3_S

<!-- ops-step
id: strata.packages
ansible_task: strata.packages | Install Strata prerequisites
kind: package
-->

```bash { name=strata-settings }
export strata_user=strata
export strata_source=/opt/strata
export strata_home=/var/lib/strata
export strata_memory_limit_mib=$(awk '/^MemTotal:/ {print int($2/1024) - 10240}' /proc/meminfo)
export STRATA_PARALLEL=2
export STRATA_CONTEXT=262144  # Tokens per user
export STRATA_IQ3_KV=q4_0
sudo apt update
sudo apt install acl build-essential git ca-certificates curl cmake ninja-build \
  python3 python3-venv python3-dev pciutils util-linux gettext-base nano
```

### 1.2 Create the service account and get the source

<!-- ops-step
id: strata.source
ansible_task: strata.source | Check out Strata and its submodules
kind: command
-->

```bash { name=strata-source }
sudo useradd --system --user-group --create-home --home-dir "$strata_home" \
  --groups render,video --shell /usr/sbin/nologin "$strata_user"
sudo install -d -o "$strata_user" -g "$strata_user" "$strata_source" /opt/strata-data
sudo -u "$strata_user" git clone --recurse-submodules \
  https://github.com/Niko1221/Strata.git "$strata_source"
```

### 1.3 Build and prepare the model

Run from the operations repository. Stop running Strata instances before building.

<!-- ops-step
id: strata.prepare
ansible_task: strata.prepare | Build Strata and prepare model weights
kind: command
-->

```bash { name=strata-prepare }
sudo systemctl stop 'strata@*.service'
sudo systemd-run --wait --pipe --collect --unit=strata-prepare-iq3_s \
  --uid="$strata_user" --gid="$strata_user" --working-directory="$strata_source" \
  --setenv="HOME=$strata_home" --property="MemoryMax=${strata_memory_limit_mib}M" \
  --property=MemorySwapMax=0 --property=LimitMEMLOCK=infinity \
  --property="SupplementaryGroups=render video" --property=RuntimeMaxSec=86400 \
  "$strata_source/setup.sh" --setup --build --yes --no-start --no-browser \
  --source huggingface --family qwen --model IQ3_S --backend hip \
  --data-dir /opt/strata-data --context "$STRATA_CONTEXT" --kv "$STRATA_IQ3_KV" \
  --parallel "$STRATA_PARALLEL" --mtp on \
  --low-ram resident --kv-streaming off --vram-reserve-mib 1024 \
  --vision no --host 127.0.0.1 --port 8080
```

In `/opt/strata/strata-iq3_s.json`, append `"--conversation-cache-mib", "2048"` to both the `args` and `user_args` arrays (create `user_args` if absent). Set `"STRATA_BATCH_MTP": "0"` inside `env`. In the adjacent `strata-iq3_s.shared-settings.json`, set `"max_tokens": 16384` in the top-level object.

```bash { name=strata-serving-settings }
sudo -u "$strata_user" nano "$strata_source/strata-iq3_s.json"
sudo -u "$strata_user" nano "$strata_source/strata-iq3_s.shared-settings.json"
```

### 1.4 Start the service

<!-- ops-step
id: strata.service
ansible_task: strata.service | Install the Strata service
kind: command
-->

```bash { name=strata-service }
envsubst < roles/strata/templates/strata@.service.j2 | sudo tee /etc/systemd/system/strata@.service >/dev/null
sudo systemctl daemon-reload
sudo systemctl enable --now strata@iq3_s
```

### 1.5 Verify the default profile

Wait for the model to load. Status should show two serving slots and a context of 262,144. The output limit in settings should be 16,384 tokens.

<!-- ops-step
id: strata.verify
ansible_task: strata.verify | Verify local API readiness
kind: command
-->

```bash { name=strata-verify }
curl -fsS http://127.0.0.1:8080/v1/status
curl -fsS http://127.0.0.1:8080/settings
curl -fsS http://127.0.0.1:8080/metrics
```

### 1.6 Connect a client

| Client | Base URL on the inference host | Model |
|---|---|---|
| Claude Code (`ANTHROPIC_BASE_URL`) | `http://127.0.0.1:8080` | `qwen3.8-flash-next-iq3_s` |
| OpenAI-compatible client | `http://127.0.0.1:8080/v1` | `qwen3.8-flash-next-iq3_s` |

For a private-network endpoint, use the server’s DNS name in place of `127.0.0.1`.

### 1.7 Enable periodic SSD trim (Optional)

Use this on storage that supports trim.

<!-- ops-step
id: strata.trim
ansible_task: strata.trim | Enable periodic SSD trim when requested
kind: command
-->

```bash { name=strata-trim }
sudo systemctl enable --now fstrim.timer
```

### 1.8 Verify restored weights (Optional)

Run after copying IQ3_S shards from another installation. Upstream setup verifies fresh downloads.

```bash { name=strata-restored-weights }
sudo -u "$strata_user" python3 - "$strata_source/strata-iq3_s.json" "$strata_source/setup.py" IQ3_S \
  < roles/strata/files/verify_model.py
```
