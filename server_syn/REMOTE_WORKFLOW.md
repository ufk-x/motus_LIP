# Remote Workflow

This workspace follows a fixed workflow:

1. Edit code locally in `~/gpufree`
2. Sync code to server
3. Run and verify on server `root@120.209.70.195:30331`

Server runtime root:
- `/root/gpufree-data`

## 1. Sync Local Changes To Server

Use the same rsync rule set as the current workspace sync:

```bash
rsync -avz \
  -e "ssh -p 30331" \
  --exclude "conda/" \
  --exclude ".Trash-0/" \
  --exclude "lost+found/" \
  --exclude "RoboTwin/assets/" \
  --exclude "RoboTwin/datasets/" \
  --exclude "RoboTwin/outputs/" \
  --exclude "RoboTwin/logs/" \
  --exclude "RoboTwin/wandb/" \
  --exclude "RoboTwin/**/__pycache__/" \
  --exclude "RoboTwin/**/*.pyc" \
  --exclude "Motus/pretrained_models/" \
  --exclude "Motus/*.whl" \
  --exclude "RoboTwin/policy/Motus/*.whl" \
  ~/gpufree/ \
  root@120.209.70.195:/root/gpufree-data/
```

Meaning:
- local code is authoritative for source files
- server keeps the large runtime assets, model weights, and conda envs

Shortcut:

```bash
~/gpufree/tools/sync_to_server.sh
```

## 2. SSH To Server

Quick login:

```bash
ssh -p 30331 root@120.209.70.195
```

Shortcut:

```bash
~/gpufree/tools/ssh_server.sh
```

Recommended bootstrap after login:

```bash
source /opt/conda/etc/profile.d/conda.sh
cd /root/gpufree-data
```

## 3. Run RoboTwin Scripts On Server

### Render self-check

```bash
ssh -p 30331 root@120.209.70.195 '
source /opt/conda/etc/profile.d/conda.sh
cd /root/gpufree-data
conda run -n RoboTwin python RoboTwin/script/test_render.py
'
```

Shortcut:

```bash
~/gpufree/tools/run_robotwin_root.sh python RoboTwin/script/test_render.py
```

### Run a RoboTwin script from RoboTwin root

Use this pattern when the script relies on `./assets`, `./task_config`, or `./data`:

```bash
ssh -p 30331 root@120.209.70.195 '
source /opt/conda/etc/profile.d/conda.sh
cd /root/gpufree-data/RoboTwin
conda run -n RoboTwin python path/to/script.py
'
```

Example:

```bash
ssh -p 30331 root@120.209.70.195 '
source /opt/conda/etc/profile.d/conda.sh
cd /root/gpufree-data/RoboTwin
conda run -n RoboTwin python script/collect_data.py place_shoe demo_clean
'
```

Shortcut:

```bash
~/gpufree/tools/run_robotwin.sh python script/collect_data.py place_shoe demo_clean
```

### Run a RoboTwin script from workspace root

Use this pattern when the script path is easier to invoke from `/root/gpufree-data`:

```bash
ssh -p 30331 root@120.209.70.195 '
source /opt/conda/etc/profile.d/conda.sh
cd /root/gpufree-data
conda run -n RoboTwin python RoboTwin/script/some_script.py
'
```

Shortcut:

```bash
~/gpufree/tools/run_robotwin_root.sh python RoboTwin/script/some_script.py
```

## 4. Run Motus Scripts On Server

```bash
ssh -p 30331 root@120.209.70.195 '
source /opt/conda/etc/profile.d/conda.sh
cd /root/gpufree-data/Motus
conda run -n motus python path/to/script.py
'
```

Shortcut:

```bash
~/gpufree/tools/run_motus.sh python path/to/script.py
```

## 5. Common Checks

### Check server conda envs

```bash
ssh -p 30331 root@120.209.70.195 '
source /opt/conda/etc/profile.d/conda.sh
conda env list
'
```

### Check a server path exists

```bash
ssh -p 30331 root@120.209.70.195 '
test -e /root/gpufree-data/RoboTwin/assets && echo exists || echo missing
'
```

### Inspect important asset sizes

```bash
ssh -p 30331 root@120.209.70.195 '
cd /root/gpufree-data/RoboTwin/assets
du -sh ./*
'
```

### Inspect expected data output tree

```bash
ssh -p 30331 root@120.209.70.195 '
cd /root/gpufree-data/RoboTwin
find data -maxdepth 3 | sort | sed -n "1,200p"
'
```

## 6. Default Rules For Future Work

- Any real execution result must come from the server, not local machine output
- Any path/resource judgment must prefer `/root/gpufree-data` over local `~/gpufree`
- RoboTwin scripts with relative paths should usually run under `/root/gpufree-data/RoboTwin`
- Sync before running if local code changed

## 7. Fast Routine

For most code changes, the routine is:

```bash
# local
cd ~/gpufree

# sync
rsync -avz \
  -e "ssh -p 30331" \
  --exclude "conda/" \
  --exclude ".Trash-0/" \
  --exclude "lost+found/" \
  --exclude "RoboTwin/assets/" \
  --exclude "RoboTwin/datasets/" \
  --exclude "RoboTwin/outputs/" \
  --exclude "RoboTwin/logs/" \
  --exclude "RoboTwin/wandb/" \
  --exclude "RoboTwin/**/__pycache__/" \
  --exclude "RoboTwin/**/*.pyc" \
  --exclude "Motus/pretrained_models/" \
  --exclude "Motus/*.whl" \
  --exclude "RoboTwin/policy/Motus/*.whl" \
  ~/gpufree/ \
  root@120.209.70.195:/root/gpufree-data/

# run on server
ssh -p 30331 root@120.209.70.195 '
source /opt/conda/etc/profile.d/conda.sh
cd /root/gpufree-data/RoboTwin
conda run -n RoboTwin python script/test_render.py
'
```

Equivalent shortcut routine:

```bash
cd ~/gpufree
~/gpufree/tools/sync_to_server.sh
~/gpufree/tools/run_robotwin.sh python script/test_render.py
```
