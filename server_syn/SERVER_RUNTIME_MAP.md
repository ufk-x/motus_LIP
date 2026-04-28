# Server Runtime Map

Server:
- Host: `root@120.209.70.195`
- SSH port: `30331`
- Project root on server: `/root/gpufree-data`

Working rule:
- Code can be edited locally in `~/gpufree`
- Real execution, validation, asset lookup, and environment checks must be done on the server

## Top Level Layout

Server root: `/root/gpufree-data`

```text
/root/gpufree-data
├── RoboTwin/        ~17G
├── Motus/           ~76G
├── conda/           ~18G
├── .cache/
├── .Trash-0/
└── lost+found/
```

Conda environments on server:
- `RoboTwin` -> `/root/gpufree-data/conda/envs/RoboTwin`
- `motus` -> `/root/gpufree-data/conda/envs/motus`

Recommended remote shell bootstrap before running Python:

```bash
source /opt/conda/etc/profile.d/conda.sh
cd /root/gpufree-data
```

## RoboTwin Map

Server path: `/root/gpufree-data/RoboTwin`

Important size overview:
- `assets/` ~16G
- `policy/` ~256M
- `envs/` ~243M
- `description/` ~3.1M
- `script/` ~228K
- `task_config/` ~32K
- `data/` currently almost empty

### `RoboTwin/assets`

Server path: `/root/gpufree-data/RoboTwin/assets`

```text
assets/
├── background_texture/   ~11G
│   ├── seen/
│   └── unseen/
├── objects/              ~4.4G
├── embodiments/          ~901M
├── files/                ~5.8M
├── _download.py
├── .cache/
├── .ipynb_checkpoints/
└── __MACOSX/
```

Notes:
- `background_texture/` is the largest runtime asset bucket and likely matters for domain randomization.
- `objects/` contains both numbered RoboTwin objects and several special asset collections.
- `embodiments/` contains robot descriptions, mesh files, and planner/collision configs.

### `RoboTwin/assets/embodiments`

Server path: `/root/gpufree-data/RoboTwin/assets/embodiments`

```text
embodiments/
├── ARX-X5/
│   ├── X5A.urdf
│   ├── config.yml
│   ├── collision_X5A.yml
│   ├── curobo.yml
│   ├── curobo_tmp.yml
│   └── meshes/
├── aloha-agilex/
│   ├── config.yml
│   ├── collision_aloha_left.yml
│   ├── collision_aloha_right.yml
│   ├── curobo_left.yml
│   ├── curobo_left_tmp.yml
│   ├── curobo_right.yml
│   ├── curobo_right_tmp.yml
│   ├── meshes/
│   ├── srdf/
│   └── urdf/
├── franka-panda/
│   ├── config.yml
│   ├── collision_franka.yml
│   ├── curobo.yml
│   ├── curobo_tmp.yml
│   ├── panda.urdf
│   ├── panda.srdf
│   └── franka_description/
├── piper/
│   ├── config.yml
│   ├── collision_piper.yml
│   ├── curobo.yml
│   ├── curobo_tmp.yml
│   ├── piper.urdf
│   ├── piper.srdf
│   ├── meshes/
│   └── urdf/
└── ur5-wsg/
    ├── config.yml
    ├── collision_wsg.yml
    ├── curobo.yml
    ├── curobo_tmp.yml
    ├── ur5_wsg_gripper.urdf
    ├── ur5.srdf
    └── meshes/
```

Practical implication:
- Embodiment selection in config must match these directory names.
- If code touches URDF/SRDF/planner loading, server-side existence under `assets/embodiments/*` is the source of truth.

### `RoboTwin/assets/objects`

Server path: `/root/gpufree-data/RoboTwin/assets/objects`

Main numbered objects:
- `001_bottle` through `120_plant`
- plus helper/special directories such as `cube`, `objaverse`, `sapien-block1`, `sapien-block2`, `vis_box`

Important special directories:

```text
objects/
├── cube/
│   ├── base.mtl
│   └── textured.obj
├── objaverse/
│   ├── README.md
│   ├── list.json
│   ├── bottle/
│   ├── bowl/
│   ├── brush/
│   ├── can/
│   ├── chip_can/
│   ├── clock/
│   ├── drinkbox/
│   ├── hammer/
│   ├── marker/
│   ├── notebook/
│   ├── plate/
│   ├── pot/
│   ├── ramen_box/
│   ├── remote/
│   ├── slipper/
│   ├── snack_box/
│   ├── snack_package/
│   ├── sneaker/
│   ├── spoon/
│   ├── steel_tape/
│   ├── tape/
│   ├── thermos/
│   ├── tissue/
│   ├── toothbrush/
│   ├── toy_car/
│   └── wallet/
├── sapien-block1/
│   └── points_info.json
├── sapien-block2/
│   └── points_info.json
└── vis_box/
    ├── base.glb
    ├── functional.glb
    └── gripper.glb
```

Practical implication:
- Object lookup code may depend on either numbered object folders or special collections under `objaverse/`.
- `points_info.json` in `sapien-block1/2` looks important for geometry/keypoint logic.
- Visualization and helper geometry may come from `vis_box/`.

### `RoboTwin/assets/files`

Server path: `/root/gpufree-data/RoboTwin/assets/files`

Known files:
- `50_tasks.gif`
- `domain_randomization.png`

These look documentation/demo oriented, not core runtime assets.

### `RoboTwin/data`

Server path: `/root/gpufree-data/RoboTwin/data`

Current observed structure:

```text
data/
└── process_stuck.py
```

Practical implication:
- The checked server currently does not have a large populated `RoboTwin/data` tree under this path.
- Default config still points `save_path: ./data`, so runtime outputs may be written here later.

### `RoboTwin/task_config`

Server path: `/root/gpufree-data/RoboTwin/task_config`

```text
task_config/
├── _camera_config.yml
├── _config_template.yml
├── _embodiment_config.yml
├── _eval_step_limit.yml
├── create_task_config.sh
├── demo_clean.yml
└── demo_randomized.yml
```

Key file roles:
- `_camera_config.yml`: named camera presets like `L515`, `Large_L515`, `D435`, `Large_D435`
- `_config_template.yml`: baseline collection/eval template
- `_embodiment_config.yml`: maps embodiment names to `./assets/embodiments/...`
- `_eval_step_limit.yml`: per-task maximum step counts
- `demo_clean.yml`: example config with clean background
- `demo_randomized.yml`: example config with background/table/light randomization enabled

Important config observations:
- Default save path is `./data`
- Default embodiment examples use `aloha-agilex`
- Domain randomization toggles depend on `assets/background_texture`
- Embodiment path mapping is relative to RoboTwin root, for example:

```yaml
aloha-agilex:
  file_path: "./assets/embodiments/aloha-agilex/"
```

## Common Remote Commands

Run RoboTwin render check on server:

```bash
ssh -p 30331 root@120.209.70.195 '
source /opt/conda/etc/profile.d/conda.sh
cd /root/gpufree-data
conda run -n RoboTwin python RoboTwin/script/test_render.py
'
```

Inspect server-side assets quickly:

```bash
ssh -p 30331 root@120.209.70.195 '
cd /root/gpufree-data/RoboTwin/assets
du -sh ./*
'
```

## Collaboration Assumptions

For this workspace, treat these as defaults:
- Local `~/gpufree` is a code-sync copy, not the full runtime artifact store
- Server `/root/gpufree-data` is the authoritative runtime layout
- Missing local directories like `RoboTwin/assets/` do not imply missing runtime resources
- Before changing path logic, verify against server paths first

## Path Dependency Map

The most important pattern in RoboTwin is:
- many scripts assume the current working directory is the RoboTwin repo root
- a large amount of code uses relative paths like `./assets/...`, `./task_config/...`, and `./data/...`
- if you run from the wrong directory, paths may silently resolve wrong even when the files exist on the server

### Global path anchors

Defined in `RoboTwin/envs/_GLOBAL_CONFIGS.py`:
- `ASSETS_PATH` -> `<RoboTwin root>/assets/`
- `EMBODIMENTS_PATH` -> `<RoboTwin root>/assets/embodiments/`
- `TEXTURES_PATH` -> `<RoboTwin root>/assets/background_texture/`
- `CONFIGS_PATH` -> `<RoboTwin root>/task_config/`
- `DESCRIPTION_PATH` -> `<RoboTwin root>/description/`

This file is the main stable source for repo-root-relative paths.

### Collection pipeline

Main entry:
- `RoboTwin/script/collect_data.py`

Path dependencies:
- reads task config from `./task_config/{task_config}.yml`
- reads embodiment map from `task_config/_embodiment_config.yml`
- embodiment map points into `./assets/embodiments/...`
- writes collection outputs under `./data/{task_name}/{task_config}/`
- writes `scene_info.json` under the same output tree
- after collection, runs instruction generation from `description/`

Practical implication:
- running collection on the server requires the full `assets/embodiments/`, `assets/objects/`, and `assets/background_texture/` trees to exist
- config `save_path: ./data` is relative to RoboTwin root, not the workspace root

### Instruction generation pipeline

Main entry:
- `RoboTwin/description/utils/generate_episode_instructions.py`

Path dependencies:
- reads task instruction templates from `description/task_instruction/{task_name}.json`
- reads task setting from `task_config/{setting}.yml`
- reads scene metadata from `{save_path}/{task_name}/{setting}/scene_info.json`
- reads object descriptions from `description/objects_description/*.json`
- writes generated per-episode instructions to `data/{task_name}/{setting}/instructions/`

Practical implication:
- `save_path` inside task config controls where instruction generation looks for `scene_info.json`
- if `save_path` changes, this script follows it
- object description generation is separate from runtime assets, but still required for natural-language instruction files

### Runtime asset loading

Most direct asset loading is in:
- `RoboTwin/envs/utils/create_actor.py`
- `RoboTwin/envs/utils/rand_create_cluttered_actor.py`
- `RoboTwin/envs/utils/transforms.py`
- `RoboTwin/script/create_object_data.py`
- `RoboTwin/script/create_messy_data.py`

Common path families found in code:
- `./assets/background_texture/{texture_id}.png`
- `./assets/objects/...`
- `./assets/objects/objaverse/list.json`
- `./assets/objects/same.json`
- `./assets/objects/cube/textured.obj`
- `./assets/embodiments/...`

Practical implication:
- `assets/background_texture/` is not optional if domain randomization with random background is enabled
- `assets/objects/` is foundational and used by many environment helpers
- `assets/objects/objaverse/` is specifically used for cluttered-table generation
- embodiment loading depends on URDF/SRDF/config/collision files under `assets/embodiments/`

### Data consumers

Several policy/data-prep scripts expect collected data under the RoboTwin tree:
- `RoboTwin/policy/RDT/scripts/process_data.py`
- `RoboTwin/policy/GO1/scripts/process_data.py`
- `RoboTwin/data/process_stuck.py`

Typical expected layout:

```text
RoboTwin/data/{task_name}/{task_config}/
├── data/
│   ├── episode0.hdf5
│   ├── episode1.hdf5
│   └── ...
├── instructions/
│   ├── episode0.json
│   ├── episode1.json
│   └── ...
├── scene_info.json
└── seed.txt
```

Practical implication:
- even though current server `RoboTwin/data/` is mostly empty, many downstream scripts assume this structure will appear after collection
- bugs involving missing training/eval inputs often reduce to this expected directory tree not being populated yet

### Environment-sensitive scripts

Server environment paths matter for:
- `conda run -n RoboTwin ...`
- `conda run -n motus ...`
- package patching in `RoboTwin/script/_install.sh`

Observed server env locations:
- `/root/gpufree-data/conda/envs/RoboTwin`
- `/root/gpufree-data/conda/envs/motus`

Practical implication:
- any script that imports SAPIEN, mplib, curobo, or task env modules should be validated on the server in the correct conda env
- local imports may succeed or fail differently because the local machine does not mirror the full server environments

### Safe execution rule

When a RoboTwin script uses any of these relative prefixes:
- `./assets`
- `./task_config`
- `./data`
- `description/...`
- `envs/...`

assume it should be run with working directory:

```bash
cd /root/gpufree-data/RoboTwin
```

If a wrapper script expects project root instead, use:

```bash
cd /root/gpufree-data
```

and then invoke the RoboTwin script explicitly from there.
