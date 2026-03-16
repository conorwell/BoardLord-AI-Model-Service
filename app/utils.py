import json
import os
import torch

_DATA_DIR = os.path.join(os.path.dirname(__file__))

# Grid dimensions (TB2 Mirror coordinate space)
# x: -64 to 64, step 4 → 33 cols
# y: 4 to 140, step 4 → 35 rows
GRID_H = 35
GRID_W = 33

# Hold role → channel index: start, hand, finish, foot
ROLE_TO_CHANNEL = {5: 0, 6: 1, 7: 2, 8: 3}
PLASTIC_CHANNEL = 4
NUM_CHANNELS = 5

PLASTIC_SET_ID = 13  # set_id=12 is wood, set_id=13 is plastic

MAX_VGRADE = 11   # V11+ is a single class
LOW_COLLAPSE = 2  # V0, V1, V2 collapsed into ≤V2
NUM_CLASSES = MAX_VGRADE - LOW_COLLAPSE + 1  # 10 classes: ≤V2, V3–V10, V11+


def load_position_map() -> dict:
    """Return dict {position_id (int) -> (row, col)} for TB2 Mirror."""
    with open(os.path.join(_DATA_DIR, "position_map.json")) as f:
        raw = json.load(f)
    pos_map = {}
    for pid_str, (x, y) in raw.items():
        col = (x + 64) // 4       # 0–32, left → right
        row = (140 - y) // 4      # 0–34, top → bottom
        pos_map[int(pid_str)] = (row, col)
    return pos_map


def load_material_map() -> dict:
    """Return dict {position_id (int) -> 1.0 if plastic, 0.0 if wood}."""
    with open(os.path.join(_DATA_DIR, "placements.json")) as f:
        placements = json.load(f)
    return {p['id']: 1.0 if p['set_id'] == PLASTIC_SET_ID else 0.0
            for p in placements}


def encode_climb(holds: list[dict], pos_map: dict, material_map: dict) -> torch.Tensor:
    """Return (NUM_CHANNELS, GRID_H, GRID_W) float tensor encoding a climb's holds."""
    grid = torch.zeros(NUM_CHANNELS, GRID_H, GRID_W, dtype=torch.float32)
    for h in holds:
        pid = h['position_id']
        ch = ROLE_TO_CHANNEL.get(h['role'])
        if ch is None or pid not in pos_map:
            continue
        row, col = pos_map[pid]
        grid[ch, row, col] = 1.0
        grid[PLASTIC_CHANNEL, row, col] = material_map.get(pid, 0.0)
    return grid


def vgrade_to_label(cls: int) -> str:
    """Convert model class index to display string."""
    cls = max(0, min(NUM_CLASSES - 1, round(cls)))
    if cls == 0:
        return f'≤V{LOW_COLLAPSE}'
    v = cls + LOW_COLLAPSE
    return 'V11+' if v >= MAX_VGRADE else f'V{v}'
