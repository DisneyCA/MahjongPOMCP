"""
Mahjong POMDP Agent with POMCP Planning

This implementation models Mahjong as a Partially Observable Markov Decision Process (POMDP)
and uses Partially Observable Monte Carlo Planning (POMCP) for Player 0's decision-making.

Key concepts from Algorithms for Decision Making (Kochenderfer et al.):
- POMDP formulation with hidden state (Ch. 19)
- Particle-based belief representation (Ch. 19, Sec. 19.6)
- Online belief state planning (Ch. 22)
- POMCP: Monte Carlo tree search for POMDPs (Ch. 22.5, Algorithm 22.1)
- Sparse sampling and rollout policies (Ch. 22)
"""

import random
import copy
import math
from collections import Counter
from dataclasses import dataclass, field
from typing import List, Optional, Dict, Tuple
import pygame


# =============================
# GAME CONSTANTS AND UI SETUP
# =============================

WINDOW_WIDTH = 1250
WINDOW_HEIGHT = 1020
BG_COLOR = (0, 100, 0)   # green table

TILE_W, TILE_H = 50, 70  # adjust to match your jpg aspect ratio

SCREEN = None
TILE_IMAGES = {}  # dict[str, pygame.Surface]
FONT = None

BAMBOOCLASS = ["1B", "2B", "3B", "4B", "5B", "6B", "7B", "8B", "9B"]  # Bamboo suit
CHARACTERCLASS = ["1C", "2C", "3C", "4C", "5C", "6C", "7C", "8C", "9C"]  # Characters suit
CIRCLECLASS = ["1D", "2D", "3D", "4D", "5D", "6D", "7D", "8D", "9D"]  # Dots suit
WINDCLASS = ["EW", "NW", "SW", "WW"]  # Winds (honors)
DRAGONCLASS = ["RD", "GD", "WD"]  # Dragons (honors)

ALL_TILES = BAMBOOCLASS + CHARACTERCLASS + CIRCLECLASS + WINDCLASS + DRAGONCLASS


# =============================
# POMDP STATE REPRESENTATION
# =============================
# POMDP state S: full Mahjong configuration including all hidden tiles.
# This represents a complete state in the POMDP tuple (S, A, O, T, R, γ) from Ch. 19.

@dataclass
class Player:
    """Represents a player's hand and public information."""
    name: str
    concealed: List[str] = field(default_factory=list)  # Hidden hand tiles
    exposed: List[List[str]] = field(default_factory=list)  # Public melds (pung/gong/chow)
    discards: List[str] = field(default_factory=list)  # Public discard history

@dataclass
class GameState:
    """
    Complete game state S for the Mahjong POMDP.
    Includes both observable (discards, exposed melds) and hidden (concealed tiles, wall) information.
    """
    wall: List[str]
    players: List[Player]
    current_player: int = 0
    last_discard: Optional[Tuple[str, int]] = None  # (tile, player_index)
    log: List[Dict] = field(default_factory=list)
    step: int = 0

@dataclass(frozen=True)
class P0Action:
    """
    P0Action encodes the POMDP action space A for Player 0.
    Actions include: DISCARD (normal play), WIN (win on discard), 
    PUNG/GONG/CHOW (claiming tiles), SELF_WIN/SELF_GONG (own turn actions), PASS.
    """
    kind: str  # Action type
    payload: object  # Action parameters (e.g., discard index, chow sequence, tile)


# =============================
# UTILITY FUNCTIONS
# =============================

def init_ui():
    """Initialize pygame window, font, and load tile images for visualization."""
    global SCREEN, TILE_IMAGES, FONT

    pygame.init()
    SCREEN = pygame.display.set_mode((WINDOW_WIDTH, WINDOW_HEIGHT))
    pygame.display.set_caption("Mahjong Viewer")

    # Basic font
    FONT = pygame.font.SysFont("arial", 24)

    # Load each tile image once and scale to TILE_W x TILE_H
    # Also create rotated versions for each player direction
    for code in ALL_TILES:
        path = f"output_tiles/{code}.jpg"
        img = pygame.image.load(path)  # [web:100]
        img = pygame.transform.smoothscale(img, (TILE_W, TILE_H))
        TILE_IMAGES[code] = img  # 0 degrees (bottom player)
        
        # Rotate 90 degrees clockwise for right player
        img_90 = pygame.transform.rotate(img, -90)
        TILE_IMAGES[f"{code}_90"] = img_90
        
        # Rotate 180 degrees for top player
        img_180 = pygame.transform.rotate(img, 180)
        TILE_IMAGES[f"{code}_180"] = img_180
        
        # Rotate 270 degrees (90 counter-clockwise) for left player
        img_270 = pygame.transform.rotate(img, 90)
        TILE_IMAGES[f"{code}_270"] = img_270

def create_wall() -> List[str]:
    # 4 copies of each tile, ignore flowers
    wall = []
    for t in ALL_TILES:
        wall.extend([t] * 4)
    random.shuffle(wall)
    return wall

def tile_suit(tile: str) -> str:
    # last character is suit / class marker
    return tile[-1]

def tile_rank(tile: str) -> Optional[int]:
    # honors (winds, dragons) have no numeric rank
    if tile in WINDCLASS or tile in DRAGONCLASS:
        return None
    return int(tile[0])

def sort_tiles(tiles: List[str]) -> List[str]:
    """Sort tiles by class (Bamboo, Character, Circle, Wind, Dragon) and numerical order."""
    def tile_sort_key(tile):
        suit = tile_suit(tile)
        # Class priority: B=0, C=1, D=2, W=3, others=4
        if suit == 'B':
            class_priority = 0
        elif suit == 'C':
            class_priority = 1
        elif suit == 'D':
            class_priority = 2
        elif suit == 'W':
            class_priority = 3
        else:  # Dragons
            class_priority = 4
        
        rank = tile_rank(tile)
        if rank is None:
            # For honors, use tile code for consistent ordering
            rank = 99 + ALL_TILES.index(tile) if tile in ALL_TILES else 999
        
        return (class_priority, rank)
    
    return sorted(tiles, key=tile_sort_key)


# =============================
# HEURISTIC VALUE FUNCTIONS
# =============================
# Approximate value functions U(b) used for reward shaping and rollout evaluation (Ch. 22).

def player_tiles_for_scoring(player) -> List[str]:
    """Return all tiles used for 4 melds + 1 pair evaluation.

    Kongs are treated as pungs (three tiles) for pattern checking.
    """
    tiles: List[str] = []
    for meld in player.exposed:
        if len(meld) == 4:
            tiles.extend(meld[:3])  # treat gong as pung
        else:
            tiles.extend(meld)
    tiles.extend(player.concealed)
    return tiles

def meld_score(tiles: List[str]) -> float:
    """
    Crude heuristic: how many melds (triplets/sequences) we can greedily extract.
    Higher is better. This is NOT exact, just for shaping.
    """
    counts = Counter(tiles)
    score = 0.0

    # Count pungs (triplets)
    for tile, c in list(counts.items()):
        if c >= 3:
            k = c // 3
            score += k
            counts[tile] -= 3 * k

    # Count chows (sequences) in each suit greedily
    for s in ("B", "C", "D"):
        # build multiset of ranks remaining in this suit
        ranks = []
        for r in range(1, 10):
            t = f"{r}{s}"
            ranks.extend([r] * counts.get(t, 0))

        ranks.sort()
        # greedy: repeatedly try to remove r,r+1,r+2
        changed = True
        while changed:
            changed = False
            for r in range(1, 8):
                if r in ranks and (r+1) in ranks and (r+2) in ranks:
                    # remove one of each
                    ranks.remove(r)
                    ranks.remove(r+1)
                    ranks.remove(r+2)
                    score += 1.0
                    changed = True
                    break

    return score

def hand_progress(player: "Player") -> float:
    """
    Heuristic value function U(b) measuring hand quality for reward shaping.
    Uses all tiles (exposed + concealed) to estimate progress toward winning.
    Used in POMCP simulations for potential-based reward shaping (Ch. 22).
    """
    tiles = player_tiles_for_scoring(player)
    return meld_score(tiles)

def hand_potential_score(tiles: List[str], state: Optional["GameState"] = None) -> int:
    """
    Detailed heuristic value function for leaf node evaluation in MCTS.
    Scores complete and partial melds, penalizes dead tiles (3+ copies seen publicly).
    Used in rollout policy and greedy discard selection.
    """
    counts = Counter(tiles)
    score = 0

    # Complete pungs (triplets)
    for t, c in counts.items():
        if c >= 3:
            score += 4  # full meld

    # Complete chows (sequences) - scan each suit
    for suit in ("B", "C", "D"):
        for r in range(1, 8):  # 1-7, can form r,r+1,r+2
            t1 = f"{r}{suit}"
            t2 = f"{r+1}{suit}"
            t3 = f"{r+2}{suit}"
            if counts.get(t1, 0) > 0 and counts.get(t2, 0) > 0 and counts.get(t3, 0) > 0:
                score += 4
                # Consume these tiles so we don't double-count
                counts[t1] -= 1
                counts[t2] -= 1
                counts[t3] -= 1

    # Near-melds: pairs
    for t, c in counts.items():
        if c == 2:
            score += 2

    # Near-melds: two consecutive suited tiles (proto-sequences)
    for suit in ("B", "C", "D"):
        for r in range(1, 9):  # 1-8, can have r and r+1
            t1 = f"{r}{suit}"
            t2 = f"{r+1}{suit}"
            if counts.get(t1, 0) > 0 and counts.get(t2, 0) > 0:
                score += 2

    # Near-melds: suited tiles with one gap (r and r+2, e.g. 3B,5B)
    for suit in ("B", "C", "D"):
        for r in range(1, 8):  # 1-7, can have r and r+2
            t1 = f"{r}{suit}"
            t3 = f"{r+2}{suit}"
            if counts.get(t1, 0) > 0 and counts.get(t3, 0) > 0:
                score += 1   # new gapped near-meld

    # Penalize single honors (winds/dragons with count == 1)
    for t, c in counts.items():
        if c == 1 and (t in WINDCLASS or t in DRAGONCLASS):
            score -= 3

    # Strong penalty for dead single honors (3+ copies seen)
    if state is not None:
        seen = Counter()
        for p in state.players:
            for tile in p.discards:
                seen[tile] += 1
            for meld in p.exposed:
                for tile in meld:
                    seen[tile] += 1
        
        for t, c in counts.items():
            if c == 1 and (t in WINDCLASS or t in DRAGONCLASS) and seen[t] >= 3:
                score -= 100  # big penalty for dead honors

    return score


# =============================
# OPPONENT MODELS (ROLLOUT POLICIES)
# =============================
# rollout_discard_for_player defines the rollout policy π used for leaf value
# estimation in MCTS (Ch. 22). Different AIs simulate diverse opponent behaviors.

def ai1_discard_random(state: "GameState", idx: int):
    """AI 1: Random discard policy."""
    p = state.players[idx]
    if not p.concealed:
        return
    k = random.randrange(len(p.concealed))
    tile = p.concealed.pop(k)
    p.discards.append(tile)
    state.last_discard = (tile, idx)
    log_action(state, idx, "discard", {"tile": tile})


def ai2_discard_wind_dragon_then_isolated(state: "GameState", idx: int):
    """AI 2: Discard single honors first, then most isolated suited tiles."""
    p = state.players[idx]
    if not p.concealed:
        return

    hand = p.concealed

    # 1) Single winds/dragons (no pair)
    counts = Counter(hand)
    candidates = [t for t in hand 
                  if t in WINDCLASS + DRAGONCLASS and counts[t] == 1]
    if candidates:
        tile = random.choice(candidates)
        hand.remove(tile)
        p.discards.append(tile)
        state.last_discard = (tile, idx)
        log_action(state, idx, "discard", {"tile": tile})
        return

    # 2) If all honors are at least pairs, pick most 'isolated' suited tile
    best_tile = None
    best_dist = -1

    # Precompute ranks by suit
    suit_to_ranks: Dict[str, List[int]] = {"B": [], "C": [], "D": []}
    for t in hand:
        r = tile_rank(t)
        s = tile_suit(t)
        if r is not None and s in suit_to_ranks:
            suit_to_ranks[s].append(r)
    for s in suit_to_ranks:
        suit_to_ranks[s].sort()

    for t in hand:
        r = tile_rank(t)
        s = tile_suit(t)

        # Honors: treat as somewhat isolated but less extreme than a lone suit
        if r is None:
            dist = 5
        else:
            ranks = suit_to_ranks[s]
            same = [x for x in ranks if x != r]
            if not same:
                dist = 10  # completely isolated in its suit
            else:
                dist = min(abs(r - x) for x in same)

        if dist > best_dist:
            best_dist = dist
            best_tile = t

    tile = best_tile
    hand.remove(tile)
    p.discards.append(tile)
    state.last_discard = (tile, idx)
    log_action(state, idx, "discard", {"tile": tile, "isolation": best_dist})


def ai3_discard_most_seen(state: "GameState", idx: int):
    """AI 3: Discard the safest tile (most frequently seen in discards/exposed melds)."""
    p = state.players[idx]
    if not p.concealed:
        return

    # Count seen tiles (discards + exposed)
    seen = Counter()
    for j, pl in enumerate(state.players):
        for t in pl.discards:
            seen[t] += 1
        for meld in pl.exposed:
            for t in meld:
                seen[t] += 1

    # Score each tile in hand by seen count
    best_tile = None
    best_seen = -1
    for t in p.concealed:
        c = seen[t]
        if c > best_seen:
            best_seen = c
            best_tile = t

    # If all unseen, pick random
    if best_tile is None:
        best_tile = random.choice(p.concealed)

    p.concealed.remove(best_tile)
    p.discards.append(best_tile)
    state.last_discard = (best_tile, idx)
    log_action(state, idx, "discard", {"tile": best_tile, "seen_count": best_seen})


def rollout_discard_for_player(state: "GameState", player_idx: int):
    """
    Rollout policy π for MCTS simulations.
    Assigns heuristic discard strategies to each player during Monte Carlo rollouts (Ch. 22).
    """
    if player_idx == 0:
        p = state.players[player_idx]
        if not p.concealed:
            return
        idx = greedy_discard_index(p, state)
        tile = p.concealed.pop(idx)
        p.discards.append(tile)
        state.last_discard = (tile, player_idx)
        return

    if player_idx == 1:
        ai1_discard_random(state, player_idx)
    elif player_idx == 2:
        ai2_discard_wind_dragon_then_isolated(state, player_idx)
    elif player_idx == 3:
        ai3_discard_most_seen(state, player_idx)
    else:
        ai1_discard_random(state, player_idx)


# =============================
# GAME LOGIC AND SIMULATOR
# =============================
# GameState + transition functions act as the generative model T,R,O used for
# sampling transitions and observations in POMCP (Ch. 22).

@dataclass(frozen=True)
class P0Action:
    """
    P0Action encodes the POMDP action space A for Player 0.
    Actions include: DISCARD (normal play), WIN (win on discard), 
    PUNG/GONG/CHOW (claiming tiles), SELF_WIN/SELF_GONG (own turn actions), PASS.
    """
    kind: str  # Action type
    payload: object  # Action parameters (e.g., discard index, chow sequence, tile)

def is_terminal_state(state: "GameState") -> bool:
    """Check if state is terminal (someone wins or wall empty)."""
    if not state.wall:
        return True
    for p in state.players:
        if is_winning_hand_with_exposed(p.concealed, p.exposed):
            return True
    return False

def winner_index(state: "GameState") -> Optional[int]:
    """Return index of winning player, or None."""
    for i, p in enumerate(state.players):
        if is_winning_hand_with_exposed(p.concealed, p.exposed):
            return i
    return None

def draw_tile_row(codes, start_x, start_y, dx, dy, rotation=0):
    """
    Draw tiles in order starting at (start_x, start_y),
    stepping by (dx, dy) per tile.
    rotation: 0, 90, 180, or 270 degrees
    """
    for i, code in enumerate(codes):
        if rotation == 0:
            img = TILE_IMAGES.get(code)
        else:
            img = TILE_IMAGES.get(f"{code}_{rotation}")
        
        if img is None:
            continue
        x = start_x + i * dx
        y = start_y + i * dy
        SCREEN.blit(img, (x, y))


# =============================
# POMDP STATE AND DATA STRUCTURES  
# =============================
# Moved earlier - see lines 30-65

# =============================
# WINNING HAND VALIDATION
# =============================

def is_winning_hand(tiles: List[str]) -> bool:
    """Check 4 melds + 1 pair, sequences only in suits; no special hands."""
    if len(tiles) != 14:
        return False
    # Count tiles by (suit, rank) or honors directly
    counts: Dict[str, int] = Counter(tiles)

    # Try every possible pair
    for tile, c in list(counts.items()):
        if c >= 2:
            counts[tile] -= 2
            if _can_form_melds(counts):
                counts[tile] += 2
                return True
            counts[tile] += 2
    return False

def is_winning_hand_with_exposed(concealed: List[str], exposed: List[List[str]]) -> bool:
    """
    Check standard hand: total 4 melds + 1 pair,
    where melds in `exposed` are already fixed and
    only `concealed` tiles can be rearranged.
    """
    # Count exposed melds that are valid (triplet or chow)
    exposed_melds = 0
    for meld in exposed:
        if len(meld) == 3 or len(meld) == 4:
            exposed_melds += 1
        else:
            return False  # ignore exotic cases for now

    # Total melds must be 4
    needed_from_concealed = 4 - exposed_melds
    if needed_from_concealed < 0:
        return False

    # Concealed tiles must be enough to form needed melds + 1 pair
    if len(concealed) != needed_from_concealed * 3 + 2:
        return False

    counts = Counter(concealed)

    # Try every possible pair in concealed
    for tile, c in list(counts.items()):
        if c >= 2:
            counts[tile] -= 2
            if _can_form_exact_melds(counts, needed_from_concealed):
                counts[tile] += 2
                return True
            counts[tile] += 2

    return False

def _can_form_exact_melds(counts: Dict[str, int], target_melds: int) -> bool:
    """
    Variant of _can_form_melds that requires exactly `target_melds` melds.
    """
    # No tiles left: success if we formed exactly target_melds
    if all(c == 0 for c in counts.values()):
        return target_melds == 0

    # If no melds left to place but still tiles, fail
    if target_melds <= 0:
        return False

    # Find first tile with count > 0
    tile = next(t for t, c in counts.items() if c > 0)

    # Try pung (triplet)
    if counts[tile] >= 3:
        counts[tile] -= 3
        if _can_form_exact_melds(counts, target_melds - 1):
            counts[tile] += 3
            return True
        counts[tile] += 3

    # Try chow (sequence) if suited tile
    r = tile_rank(tile)
    s = tile_suit(tile)
    if r is not None and s in ("B", "C", "D"):
        if r <= 7:
            t1 = f"{r}{s}"
            t2 = f"{r+1}{s}"
            t3 = f"{r+2}{s}"
            if counts.get(t1, 0) > 0 and counts.get(t2, 0) > 0 and counts.get(t3, 0) > 0:
                counts[t1] -= 1
                counts[t2] -= 1
                counts[t3] -= 1
                if _can_form_exact_melds(counts, target_melds - 1):
                    counts[t1] += 1
                    counts[t2] += 1
                    counts[t3] += 1
                    return True
                counts[t1] += 1
                counts[t2] += 1
                counts[t3] += 1

    return False

def _can_form_melds(counts: Dict[str, int]) -> bool:
    # If all counts are zero, success
    if all(c == 0 for c in counts.values()):
        return True

    # Find first tile with count > 0
    tile = next(t for t, c in counts.items() if c > 0)

    # Try pung (triplet)
    if counts[tile] >= 3:
        counts[tile] -= 3
        if _can_form_melds(counts):
            counts[tile] += 3
            return True
        counts[tile] += 3

    # Try chow (sequence) if suited tile
    r = tile_rank(tile)
    s = tile_suit(tile)
    if r is not None and s in ("B", "C", "D"):
        # Need r, r+1, r+2 of same suit
        if r <= 7:
            t1 = f"{r}{s}"
            t2 = f"{r+1}{s}"
            t3 = f"{r+2}{s}"
            if counts.get(t1, 0) > 0 and counts.get(t2, 0) > 0 and counts.get(t3, 0) > 0:
                counts[t1] -= 1
                counts[t2] -= 1
                counts[t3] -= 1
                if _can_form_melds(counts):
                    counts[t1] += 1
                    counts[t2] += 1
                    counts[t3] += 1
                    return True
                counts[t1] += 1
                counts[t2] += 1
                counts[t3] += 1

    return False


# =============================
# VISUALIZATION
# =============================

def flat_exposed(player: Player) -> List[str]:
    """Flatten all exposed melds into a single list of tiles."""
    return [t for meld in player.exposed for t in meld]

def render_state(state: GameState):
    """Graphical display of exposed melds (around table) and discards (center)."""
    SCREEN.fill(BG_COLOR)

    # --- Top: tiles remaining in wall ---
    text_surf = FONT.render(f"Tiles remaining in wall: {len(state.wall)}", True, (255, 255, 255))
    SCREEN.blit(text_surf, (20, 10))

    # --- Center: discards grid (auto-sized & centered) ---
    # Sort discards chronologically using the game log
    all_discards = []
    for entry in state.log:
        if entry["action"] == "discard":
            all_discards.append(entry["info"]["tile"])

    if all_discards:
        n = len(all_discards)

        # Fixed width (columns), variable height (rows)
        max_cols = 14
        cols = min(max_cols, n)

        # Number of rows needed
        rows = math.ceil(n / cols)

        # Total grid size in pixels
        total_w = cols * TILE_W
        total_h = rows * TILE_H

        # Center the whole discard rectangle on the screen
        start_x = (WINDOW_WIDTH - total_w) // 2
        start_y = (WINDOW_HEIGHT - total_h) // 2

        for i, code in enumerate(all_discards):
            row = i // cols  # Fill row by row (left to right, top to bottom)
            col = i % cols
            x = start_x + col * TILE_W
            y = start_y + row * TILE_H
            img = TILE_IMAGES.get(code)
            if img:
                SCREEN.blit(img, (x, y))



    # ---------- Player 0: bottom (horizontal, facing up) ----------
    p0 = state.players[0]
    bottom_concealed_y = WINDOW_HEIGHT - TILE_H - 80      # move up from edge
    bottom_exposed_y = bottom_concealed_y - TILE_H - 30   # more gap to avoid overlap

    # Concealed (sorted)
    sorted_concealed_p0 = sort_tiles(p0.concealed)
    concealed_x0 = (WINDOW_WIDTH - len(sorted_concealed_p0) * TILE_W) // 2
    draw_tile_row(sorted_concealed_p0, concealed_x0, bottom_concealed_y, TILE_W, 0, rotation=0)

    # Exposed (flattened, contiguous)
    flat0 = flat_exposed(p0)
    exposed_x0 = (WINDOW_WIDTH - len(flat0) * TILE_W) // 2
    draw_tile_row(flat0, exposed_x0, bottom_exposed_y, TILE_W, 0, rotation=0)

    name_surf = FONT.render(f"{p0.name} (POMCP)", True, (255, 255, 255))
    SCREEN.blit(name_surf, (concealed_x0, bottom_concealed_y + TILE_H + 10))

    # ---------- Player 2: top (horizontal, facing down) ----------
    p2 = state.players[2]
    top_concealed_y = 80
    top_exposed_y = top_concealed_y + TILE_H + 30

    sorted_concealed_p2 = sort_tiles(p2.concealed)
    concealed_x2 = (WINDOW_WIDTH - len(sorted_concealed_p2) * TILE_W) // 2
    draw_tile_row(sorted_concealed_p2, concealed_x2, top_concealed_y, TILE_W, 0, rotation=180)

    flat2 = flat_exposed(p2)
    exposed_x2 = (WINDOW_WIDTH - len(flat2) * TILE_W) // 2
    draw_tile_row(flat2, exposed_x2, top_exposed_y, TILE_W, 0, rotation=180)

    name_surf = FONT.render(p2.name, True, (255, 255, 255))
    SCREEN.blit(name_surf, (concealed_x2, top_concealed_y - 30))

    # ---------- Player 1: right (vertical, facing right) ----------
    p1 = state.players[1]
    right_concealed_x = WINDOW_WIDTH - TILE_H - 80   # more margin from edge
    right_exposed_x = right_concealed_x - TILE_H - 30

    sorted_concealed_p1 = sort_tiles(p1.concealed)
    # vertical: spacing by TILE_W because tiles are rotated
    concealed_y1 = (WINDOW_HEIGHT - len(sorted_concealed_p1) * TILE_W) // 2
    draw_tile_row(sorted_concealed_p1, right_concealed_x, concealed_y1, 0, TILE_W, rotation=270)

    flat1 = flat_exposed(p1)
    exposed_y1 = (WINDOW_HEIGHT - len(flat1) * TILE_W) // 2
    draw_tile_row(flat1, right_exposed_x, exposed_y1, 0, TILE_W, rotation=270)

    name_surf = FONT.render(p1.name, True, (255, 255, 255))
    SCREEN.blit(name_surf, (right_concealed_x + TILE_W + 10, concealed_y1))

    # ---------- Player 3: left (vertical, facing left) ----------
    p3 = state.players[3]
    left_concealed_x = 80
    left_exposed_x = left_concealed_x + TILE_H + 30

    sorted_concealed_p3 = sort_tiles(p3.concealed)
    concealed_y3 = (WINDOW_HEIGHT - len(sorted_concealed_p3) * TILE_W) // 2
    draw_tile_row(sorted_concealed_p3, left_concealed_x, concealed_y3, 0, TILE_W, rotation=90)

    flat3 = flat_exposed(p3)
    exposed_y3 = (WINDOW_HEIGHT - len(flat3) * TILE_W) // 2
    draw_tile_row(flat3, left_exposed_x, exposed_y3, 0, TILE_W, rotation=90)

    name_surf = FONT.render(p3.name, True, (255, 255, 255))
    SCREEN.blit(name_surf, (left_concealed_x - 80, concealed_y3))

    # ---------- Current player label ----------
    cur_name = state.players[state.current_player].name
    cp_surf = FONT.render(f"Current: {cur_name}", True, (255, 255, 0))
    SCREEN.blit(cp_surf, (20, WINDOW_HEIGHT - 40))

    pygame.display.flip()



def log_action(state: GameState, player_idx: int, action: str, info: Dict):
    """Log game actions for replay and debugging."""
    entry = {
        "step": state.step,
        "player": player_idx,
        "name": state.players[player_idx].name,
        "action": action,
        "info": info,
    }
    state.log.append(entry)


# =============================
# BELIEF STATE REPRESENTATION
# =============================
# Particle-based belief representation over hidden game states (Ch. 19, Sec. 19.6).
# MahjongBelief holds particles consistent with Player 0's observations.

class MahjongBelief:
    """
    Particle-based belief representation for POMDP (Ch. 19, particle filters).
    Each particle is a complete GameState consistent with Player 0's observations
    (own hand, all discards, all exposed melds) but differing in hidden tiles.
    """
    def __init__(self, particles: List["GameState"]):
        self.particles = particles

    def sample_state(self) -> "GameState":
        """Sample a random belief particle for POMCP simulation."""
        return copy.deepcopy(random.choice(self.particles))


def build_belief_from_public_state(real_state: "GameState", num_particles: int = 64) -> MahjongBelief:
    """
    Belief initialization: sample particles consistent with public information (Ch. 19).
    Each particle represents a possible full state where:
    - Player 0's concealed/exposed tiles and all public info are fixed
    - Opponents' concealed tiles and wall tiles are randomly sampled from remaining pool
    """
    particles = []

    # Count total copies in the real game (4 of each tile)
    total_counts = Counter({t: 4 for t in ALL_TILES})

    # Subtract all *known* tiles from total: P0 concealed + everyone's discards/exposed
    known_tiles = []

    # P0 concealed and exposed melds
    p0 = real_state.players[0]
    known_tiles.extend(p0.concealed)
    for meld in p0.exposed:
        known_tiles.extend(meld)

    # Everyone's discards and exposed melds
    # Avoid double-counting P0's exposed by only looping players 1–3 here.
    for p in real_state.players[1:]:
        known_tiles.extend(p.discards)
        for meld in p.exposed:
            known_tiles.extend(meld)

    known_counts = Counter(known_tiles)
    for t, c in known_counts.items():
        total_counts[t] -= c
        if total_counts[t] < 0:
            total_counts[t] = 0  # safety

    # Remaining tile multiset is what can be hidden (opponents + wall)
    remaining_pool = []
    for t, c in total_counts.items():
        remaining_pool.extend([t] * c)

    for _ in range(num_particles):
        rem = remaining_pool.copy()
        random.shuffle(rem)

        # New particle state
        players = [Player(name=f"Player {i}") for i in range(4)]
        wall = []

        # Copy P0 exactly (what P0 knows)
        players[0].concealed = p0.concealed.copy()
        players[0].exposed = copy.deepcopy(p0.exposed)
        players[0].discards = p0.discards.copy()

        # For other players, reassign concealed tiles *only* for unknown part
        # 1) Figure out how many concealed tiles they should have
        concealed_sizes = []
        for i in range(4):
            concealed_sizes.append(len(real_state.players[i].concealed))

        idx = 0
        for pid in range(1, 4):
            needed = concealed_sizes[pid]
            # We only assign tiles for the portion that is actually hidden from P0.
            # Any tiles in their exposed melds/discards are already accounted for.
            for _ in range(needed):
                if idx >= len(rem):
                    break
                players[pid].concealed.append(rem[idx])
                idx += 1
            # Copy public info
            players[pid].exposed = copy.deepcopy(real_state.players[pid].exposed)
            players[pid].discards = real_state.players[pid].discards.copy()

        # Remaining tiles go to the wall (in random order)
        wall = rem[idx:]

        particle = GameState(
            wall=wall,
            players=players,
            current_player=real_state.current_player,
            last_discard=real_state.last_discard,
            step=real_state.step,
        )
        particles.append(particle)

    return MahjongBelief(particles)


# =============================
# GREEDY DISCARD HEURISTIC
# =============================

def random_discard_sim(state: "GameState", player_idx: int):
    """Simple random discard for lightweight simulations."""
    player = state.players[player_idx]
    if not player.concealed:
        return
    idx = random.randrange(len(player.concealed))
    tile = player.concealed.pop(idx)
    player.discards.append(tile)
    state.last_discard = (tile, player_idx)

def greedy_discard_index(player: Player, state: Optional["GameState"] = None) -> int:
    """
    Greedy discard selection for rollout policy.
    Chooses discard minimizing harm to hand structure using hand_potential_score.
    Strongly favors discarding dead tiles (3+ copies visible publicly).
    """
    best_idx = 0
    best_score = -1e9

    base_tiles = player.concealed.copy()
    base_score = hand_potential_score(base_tiles, state)
    counts = Counter(base_tiles)

    # Build seen counter for dead honor detection
    seen = Counter()
    if state is not None:
        for p in state.players:
            for tile in p.discards:
                seen[tile] += 1
            for meld in p.exposed:
                for tile in meld:
                    seen[tile] += 1

    for i, t in enumerate(player.concealed):
        tmp_tiles = base_tiles.copy()
        # Remove one occurrence of t from tmp_tiles
        tmp_tiles.remove(t)
        sc = hand_potential_score(tmp_tiles, state)

        # Penalize breaking obvious complete pungs/chows
        penalty = 0
        if counts[t] >= 3:
            penalty -= 3  # part of a pung
        # simple chow membership check
        r = tile_rank(t)
        s = tile_suit(t)
        if r is not None and s in ("B", "C", "D"):
            for dr in (-2, -1, 0):
                base = r + dr
                if 1 <= base <= 7:
                    seq = [f"{base}{s}", f"{base+1}{s}", f"{base+2}{s}"]
                    if all(base_tiles.count(x) > 0 for x in seq):
                        penalty -= 3
                        break

        total = sc + penalty

        # Bonus for discarding single honors
        if counts[t] == 1 and (t in WINDCLASS or t in DRAGONCLASS):
            total += 1
            # HUGE bonus for discarding dead honors (3+ copies seen)
            if state is not None and seen[t] >= 3:
                total += 10

        if total > best_score:
            best_score = total
            best_idx = i

    return best_idx


# =============================
# MONTE CARLO ROLLOUTS
# =============================
# simulate_random_game: finite-horizon forward search with Monte Carlo rollouts
# (lookahead with rollouts, Ch. 22).

def simulate_random_game(state: "GameState", max_steps: int = 300) -> float:
    """
    Monte Carlo rollout from current state to terminal or depth limit.
    Returns immediate reward: +1.0 if P0 wins, -0.25 if P0 deals into opponent win, 0 otherwise.
    Uses rollout policies (AI heuristics) without call actions for speed.
    """
    steps = 0
    while steps < max_steps and not is_terminal_state(state):
        pidx = state.current_player

        if not state.wall:
            break

        # Draw
        draw_tile(state, pidx)

        # Auto-win in rollout if hand complete:
        # use concealed + exposed, not a flattened 14-tile multiset.
        p = state.players[pidx]
        if is_winning_hand_with_exposed(p.concealed, p.exposed):
            return 1.0 if pidx == 0 else 0.0

        # Discard according to rollout policy (heuristic per player)
        rollout_discard_for_player(state, pidx)

        # After a discard, allow simple win for opponents in rollouts
        if state.last_discard:
            tile, discarder = state.last_discard
            # Check each other player for win
            for offset in range(1, 4):
                pid = (discarder + offset) % 4
                if pid == discarder:
                    continue
                opp = state.players[pid]
                test_concealed = opp.concealed.copy()
                test_concealed.append(tile)
                if is_winning_hand_with_exposed(test_concealed, opp.exposed):
                    # Opponent wins on discard.
                    # If P0 fed this win, P0 gets -1; otherwise 0.
                    if pid == 0:
                        return 1.0  # P0 wins in rollout
                    elif discarder == 0:
                        return -0.25  # P0 dealt into someone else's win
                    else:
                        return 0.0   # two non-P0 players interacted

        # Ignore chow/pung/gong in rollouts to keep them fast and simple

        # Next player
        state.current_player = next_player_idx(pidx)
        steps += 1

    return 0.0


# =============================
# POMCP: MONTE CARLO TREE SEARCH FOR POMDPs
# =============================
# Implementation of POMCP (Ch. 22.5, Algorithm 22.1).
# MCTSNode stores N(h,a) and Q(h,a) for a history tree, where h represents
# an action-observation history from Player 0's perspective.

class MCTSNode:
    """
    Node in the POMCP search tree.
    Stores visit counts N(h,a) and action values Q(h,a) for UCB selection (Ch. 22.5).
    """
    def __init__(self):
        self.N = 0  # Total visits to this history node
        self.N_a: Dict = {}  # Visit count per action: N(h,a)
        self.Q_a: Dict = {}  # Action value: Q(h,a)
        self.children: Dict = {}  # Child nodes indexed by (action, observation)


class Player0Planner:
    """
    POMCP planner for Player 0 (Ch. 22.5, Algorithm 22.1).
    Online belief state planning: plans from current belief without precomputing global policy.
    Uses game simulator as generative model T,R,O(s,a) to sample next states and observations.
    """
    def __init__(self, num_simulations=3000, max_depth=8, c=1.4, shaping_alpha=0.15):
        """
        Initialize POMCP planner.
        num_simulations: Number of MCTS simulations per decision
        max_depth: Maximum search depth before rollout
        c: UCB exploration constant
        shaping_alpha: Reward shaping coefficient for hand_progress
        """
        self.num_simulations = num_simulations
        self.max_depth = max_depth
        self.c = c
        self.shaping_alpha = shaping_alpha
        self.root = MCTSNode()

    def get_actions(self, state: "GameState", reacting_to_discard: bool) -> List[P0Action]:
        """Generate all legal actions for Player 0 in current situation."""
        p0 = state.players[0]

        # Case 1: own turn (just drew)
        if not reacting_to_discard:
            actions = []

            # Self-win (tsumo)
            if is_winning_hand_with_exposed(p0.concealed, p0.exposed):
                actions.append(P0Action("SELF_WIN", None))

            # Concealed gong options
            counts = Counter(p0.concealed)
            for tile, c in counts.items():
                if c == 4:
                    actions.append(P0Action("SELF_GONG", tile))

            # Discard choices
            for idx in range(len(p0.concealed)):
                actions.append(P0Action("DISCARD", idx))

            return actions

        # Case 2: reacting to last discard from someone else
        if not state.last_discard:
            return [P0Action("PASS", None)]
        
        tile, discarder = state.last_discard
        actions = []
        base_hp = hand_progress(p0)

        # Win (win on discard): add the discard to concealed and test
        test_concealed = p0.concealed.copy()
        test_concealed.append(tile)
        if is_winning_hand_with_exposed(test_concealed, p0.exposed):
            actions.append(P0Action("WIN", tile))


        # Pung/gong options
        c = p0.concealed.count(tile)
        if c >= 2:
            actions.append(P0Action("PUNG", tile))
        if c >= 3:
            actions.append(P0Action("GONG", tile))

        # Chow options (only if Player 0 is next in turn) - with pruning
        if next_player_idx(discarder) == 0:
            chi_seqs = chow_options(p0, tile)
            for seq in chi_seqs:
                # Estimate effect on structure
                tmp = Player(name="tmp",
                            concealed=p0.concealed.copy(),
                            exposed=copy.deepcopy(p0.exposed))
                # Apply chow locally
                for t in seq:
                    if t != tile:
                        tmp.concealed.remove(t)
                tmp.exposed.append(seq.copy())
                # Only evaluate concealed tiles since exposed melds are fixed
                hp_after = meld_score(tmp.concealed)
                base_hp_concealed = meld_score(p0.concealed)
                # Only keep chow if it doesn't hurt concealed structure
                if hp_after >= base_hp_concealed:
                    actions.append(P0Action("CHOW", tuple(seq)))  # make hashable

        # Always allow PASS
        actions.append(P0Action("PASS", None))

        return actions

    def apply_action_get_obs(self, state: "GameState", action: P0Action, reacting: bool) -> Tuple[float, bool, str]:
        """
        Apply action and simulate until P0's next decision point.
        Returns (reward, terminal, observation_string).
        Implements the generative model transition T,R,O(s,a) for POMCP.
        """
        p0 = state.players[0]
        
        # Potential-based reward shaping: hand quality before action
        hp_before = hand_progress(p0)
        
        # Terminal actions
        if action.kind == "SELF_WIN" or action.kind == "WIN":
            return 1.0, True, "WIN"
        
        immediate_reward = 0.0
        
        # Apply P0's action
        if action.kind == "DISCARD":
            idx = action.payload
            tile = p0.concealed.pop(idx)
            p0.discards.append(tile)
            state.last_discard = (tile, 0)
            state.current_player = next_player_idx(0)
            
        elif action.kind == "PUNG":
            tile = action.payload
            for _ in range(2):
                p0.concealed.remove(tile)
            p0.exposed.append([tile] * 3)
            # P0 must discard after pung (use greedy heuristic)
            if p0.concealed:
                idx = greedy_discard_index(p0, state)
                disc_tile = p0.concealed.pop(idx)
                p0.discards.append(disc_tile)
                state.last_discard = (disc_tile, 0)
            state.current_player = next_player_idx(0)
            
        elif action.kind == "CHOW":
            seq = action.payload
            disc_tile, _ = state.last_discard
            for t in seq:
                if t != disc_tile:
                    p0.concealed.remove(t)
            p0.exposed.append(list(seq))
            # P0 must discard after chow (use greedy heuristic)
            if p0.concealed:
                idx = greedy_discard_index(p0, state)
                disc_tile = p0.concealed.pop(idx)
                p0.discards.append(disc_tile)
                state.last_discard = (disc_tile, 0)
            state.current_player = next_player_idx(0)
            
        elif action.kind == "GONG":
            tile = action.payload
            for _ in range(3):
                p0.concealed.remove(tile)
            p0.exposed.append([tile] * 4)
            draw_tile(state, 0)
            # P0 must discard after gong (use greedy heuristic)
            if p0.concealed:
                idx = greedy_discard_index(p0, state)
                disc_tile = p0.concealed.pop(idx)
                p0.discards.append(disc_tile)
                state.last_discard = (disc_tile, 0)
            state.current_player = next_player_idx(0)
            
        elif action.kind == "SELF_GONG":
            tile = action.payload
            for _ in range(4):
                p0.concealed.remove(tile)
            p0.exposed.append([tile] * 4)
            draw_tile(state, 0)
            state.current_player = 0
            # P0 will take another turn
            
        elif action.kind == "PASS":
            # Others continue
            if state.last_discard:
                state.current_player = next_player_idx(state.last_discard[1])
            else:
                state.current_player = next_player_idx(0)
        
        # --- Reward shaping based on change in P0's hand structure ---
        hp_after = hand_progress(p0)
        immediate_reward += self.shaping_alpha * (hp_after - hp_before)
        
        # Simulate other players until P0's next turn
        obs_events = []
        steps = 0
        max_steps = 10
        
        while state.current_player != 0 and steps < max_steps and not is_terminal_state(state):
            pidx = state.current_player
            
            if not state.wall:
                break
            
            # Opponent draws
            draw_tile(state, pidx)
            
            # Check if opponent wins
            opp = state.players[pidx]
            if is_winning_hand_with_exposed(opp.concealed, opp.exposed):
                return 0.0, True, "OPP_WIN"
            
            # Opponent discards using same heuristic policies as the real game (no calls in rollout)
            rollout_discard_for_player(state, pidx)
            if state.last_discard:
                obs_events.append(f"P{pidx}_DISC_{state.last_discard[0]}")
                
                # Check for win by other players on this discard
                tile, discarder = state.last_discard
                for offset in range(1, 4):
                    pid = (discarder + offset) % 4
                    if pid == discarder:
                        continue
                    opp = state.players[pid]
                    test_concealed = opp.concealed.copy()
                    test_concealed.append(tile)
                    if is_winning_hand_with_exposed(test_concealed, opp.exposed):
                        # Opponent wins on discard; if P0 was the discarder, this is a -1.
                        reward = -1.0 if discarder == 0 else 0.0
                        return reward, True, "OPP_RON"
            
            state.current_player = next_player_idx(pidx)
            steps += 1
        
        # Build observation string
        obs = f"STEPS:{steps}"
        if state.current_player == 0:
            obs += "_P0TURN"
        
        terminal = is_terminal_state(state)
        if terminal:
            w = winner_index(state)
            return (1.0 if w == 0 else 0.0), True, "TERMINAL"
        
        return immediate_reward, terminal, obs

    def select_ucb(self, node: MCTSNode, actions: List[P0Action]):
        """
        UCB action selection: Q(h,a) + c * sqrt(log N(h) / N(h,a)).
        Balances exploitation (Q value) and exploration (visit count) as in MCTS (Ch. 22).
        """
        best_a = None
        best_val = -1e9
        for a in actions:
            Na = node.N_a.get(a, 0)
            Qa = node.Q_a.get(a, 0.0)
            if Na == 0:
                val = float("inf")
            else:
                val = Qa + self.c * math.sqrt(math.log(node.N + 1e-9) / Na)

            # Progressive bias: prefer immediate wins very slightly
            if a.kind in ("SELF_WIN", "WIN"):
                val += 0.1

            if val > best_val:
                best_val = val
                best_a = a
        return best_a

    def simulate(self, state: "GameState", node: MCTSNode, depth: int, reacting: bool) -> float:
        """
        POMCP simulation: recursive MCTS with action-observation branching (Ch. 22.5).
        - At max depth or terminal: run Monte Carlo rollout with hand_progress shaping
        - Otherwise: select action via UCB, apply it, get observation, recurse on child node
        - Backup: update N(h,a) and Q(h,a) along the search path
        """
        if depth >= self.max_depth or is_terminal_state(state):
            # Leaf node: rollout value + heuristic shaping
            v = simulate_random_game(state)
            p0 = state.players[0]
            v += self.shaping_alpha * hand_progress(p0)
            return v

        actions = self.get_actions(state, reacting)
        if not actions:
            return 0.0

        # Initialize node on first visit
        if node.N == 0:
            for a in actions:
                node.N_a[a] = 0
                node.Q_a[a] = 0.0

        # Select action with UCB
        action = self.select_ucb(node, actions)

        # Apply action and get observation
        reward, terminal, obs = self.apply_action_get_obs(state, action, reacting)

        if terminal:
            q = reward
        else:
            # Get or create child node for this (action, observation) pair
            child_key = (action, obs)
            if child_key not in node.children:
                node.children[child_key] = MCTSNode()
            child_node = node.children[child_key]
            
            # Recurse - next step is P0's turn again if obs indicates that
            next_reacting = False  # simplified for now
            q = reward + self.simulate(state, child_node, depth + 1, next_reacting)

        # Backup
        node.N += 1
        node.N_a[action] += 1
        node.Q_a[action] += (q - node.Q_a[action]) / node.N_a[action]

        return q

    def plan_turn(self, belief: MahjongBelief, real_state: "GameState", reacting_to_discard: bool = False):
        """
        POMCP planning (Ch. 22.5, Algorithm 22.1).
        Samples belief particles, runs MCTS simulations from each, builds search tree.
        Returns best action at root based on Q-values.
        """
        # Reset root for new decision
        self.root = MCTSNode()
        
        for _ in range(self.num_simulations):
            # POMCP key step: sample a state from belief particles
            sim_state = belief.sample_state()
            
            # Override P0's public info to match reality
            sim_state.current_player = real_state.current_player
            sim_state.players[0].concealed = real_state.players[0].concealed.copy()
            sim_state.players[0].exposed = copy.deepcopy(real_state.players[0].exposed)
            sim_state.players[0].discards = real_state.players[0].discards.copy()
            sim_state.last_discard = real_state.last_discard
            # Copy all global discards/exposed melds
            for i in range(1, 4):
                sim_state.players[i].discards = real_state.players[i].discards.copy()
                sim_state.players[i].exposed = copy.deepcopy(real_state.players[i].exposed)

            # Run simulation from this sampled state
            self.simulate(sim_state, self.root, depth=0, reacting=reacting_to_discard)

        # Return best action at root
        actions = self.get_actions(real_state, reacting_to_discard)
        if not actions:
            return None

        best_a = None
        best_q = -1e9
        for a in actions:
            q = self.root.Q_a.get(a, 0.0)
            if q > best_q:
                best_q = q
                best_a = a
        
        # Debug output
        print(f"\n[AI Planning] Evaluated {len(actions)} actions, best Q={best_q:.3f}")
        return best_a


# Global planner and belief instances
PLAYER0_PLANNER = None
PLAYER0_BELIEF = None


# =============================
# GAME FLOW HELPERS
# =============================

def next_player_idx(idx: int) -> int:
    return (idx + 1) % 4

def draw_tile(state: GameState, player_idx: int) -> Optional[str]:
    if not state.wall:
        return None
    tile = state.wall.pop()
    state.players[player_idx].concealed.append(tile)
    log_action(state, player_idx, "draw", {"tile": tile})
    return tile


# =============================
# CALL DETECTION AND EXECUTION
# =============================

def pung_gong_options(player: Player, tile: str) -> List[str]:
    c = player.concealed.count(tile)
    opts = []
    if c >= 2:
        opts.append("pung")
    if c >= 3:
        opts.append("gong")
    return opts

def chow_options(player: Player, tile: str) -> List[List[str]]:
    """Return list of possible chow melds (including the discard)."""
    r = tile_rank(tile)
    s = tile_suit(tile)
    if r is None or s not in ("B", "C", "D"):
        return []
    options = []
    # sequences where tile is first, middle, or last
    ranks = [r-2, r-1, r, r+1, r+2]
    for base in (r-2, r-1, r):
        if 1 <= base <= 7:
            seq = [f"{base}{s}", f"{base+1}{s}", f"{base+2}{s}"]
            if tile not in seq:
                continue
            # need the other two tiles in hand
            needed = [t for t in seq if t != tile]
            if all(player.concealed.count(t) >= 1 for t in needed):
                options.append(seq)
    return options


# =============================
# APPLYING MELD CALLS
# =============================

def apply_pung(state: GameState, caller_idx: int, tile: str, discarder_idx: int):
    player = state.players[caller_idx]
    # remove 2 from concealed
    for _ in range(2):
        player.concealed.remove(tile)
    player.exposed.append([tile, tile, tile])
    # Remove the tile from discarder's discard pile since it was claimed
    if state.players[discarder_idx].discards and state.players[discarder_idx].discards[-1] == tile:
        state.players[discarder_idx].discards.pop()
    log_action(state, caller_idx, "pung", {"meld": [tile, tile, tile]})

def apply_gong(state: GameState, caller_idx: int, tile: str, discarder_idx: int):
    player = state.players[caller_idx]
    # remove 3 from concealed
    for _ in range(3):
        player.concealed.remove(tile)
    player.exposed.append([tile, tile, tile, tile])
    # Remove the tile from discarder's discard pile since it was claimed
    if state.players[discarder_idx].discards and state.players[discarder_idx].discards[-1] == tile:
        state.players[discarder_idx].discards.pop()
    log_action(state, caller_idx, "gong", {"meld": [tile]*4})
    # Simple supplement draw after gong
    draw_tile(state, caller_idx)

def apply_chow(state: GameState, caller_idx: int, seq: List[str], tile: str, discarder_idx: int):
    player = state.players[caller_idx]
    for t in seq:
        if t == tile:
            continue
        player.concealed.remove(t)
    player.exposed.append(seq.copy())
    # Remove the tile from discarder's discard pile since it was claimed
    if state.players[discarder_idx].discards and state.players[discarder_idx].discards[-1] == tile:
        state.players[discarder_idx].discards.pop()
    log_action(state, caller_idx, "chow", {"meld": seq})


# =============================
# HANDLING CALLS AFTER DISCARD
# =============================

def ai_wants_pung_gong(player_idx: int, opts: List[str]) -> Optional[str]:
    """Determine if AI player wants to call pung/gong (prefer gong if both available)."""
    if player_idx in (1, 2, 3):
        if "gong" in opts:
            return "gong"
        if "pung" in opts:
            return "pung"
    return None

def handle_calls_after_discard(state: GameState, discarder_idx: int, tile: str) -> int:
    """
    Process potential meld calls (pung/gong/chow) after a discard.
    Priority: win > pung/gong > chow.
    Returns next player index, or -1 if game ends.
    """
    global PLAYER0_PLANNER, PLAYER0_BELIEF
    
    n_players = len(state.players)
    
    # --- First, give Player 0 a chance to act on this discard via the planner ---
    if PLAYER0_PLANNER is not None and discarder_idx != 0:
        # Check if Player 0 has any meaningful reaction options
        p0 = state.players[0]
        has_meaningful_action = False
        
        # Check for win on discard
        test_concealed = p0.concealed.copy()
        test_concealed.append(tile)
        if is_winning_hand_with_exposed(test_concealed, p0.exposed):
            has_meaningful_action = True
        
        # Check for pung/gong
        c = p0.concealed.count(tile)
        if c >= 2:
            has_meaningful_action = True
        
        # Check for chow (only if Player 0 is next)
        if next_player_idx(discarder_idx) == 0:
            chi_opts = chow_options(p0, tile)
            if chi_opts:
                has_meaningful_action = True
        
        # Only invoke planner if there's a meaningful choice
        if has_meaningful_action:
            # Refresh belief based on current public info including this discard
            PLAYER0_BELIEF = build_belief_from_public_state(state, num_particles=32)
            p0_action = PLAYER0_PLANNER.plan_turn(PLAYER0_BELIEF, state, reacting_to_discard=True)

            if p0_action is not None and p0_action.kind != "PASS":
                print(f"\nPlayer 0 (AI) chooses reaction: {p0_action.kind} on {tile}")
                # Execute chosen reaction on the *real* state
                if p0_action.kind == "WIN":
                    # Win on discard (concealed + discard only)
                    player = state.players[0]
                    test_concealed = p0.concealed.copy()
                    test_concealed.append(tile)
                    player.concealed.append(tile)
                    log_action(state, 0, "win", {
                        "hand": sort_tiles(test_concealed),
                        "winning_tile": tile
                    })
                    print(f"{player.name} (AI) wins on discard with hand:", " ".join(sort_tiles(test_concealed)))
                    # Remove tile from discarder
                    if state.players[discarder_idx].discards and state.players[discarder_idx].discards[-1] == tile:
                        state.players[discarder_idx].discards.pop()
                    return -1

                elif p0_action.kind == "PUNG":
                    apply_pung(state, 0, tile, discarder_idx)
                    print(f"Player 0 (AI) calls pung on {tile}")
                    render_state(state)
                    # AI discards after call
                    ai_discard_for_player0(state)
                    new_tile, new_discarder = state.last_discard
                    return handle_calls_after_discard(state, new_discarder, new_tile)

                elif p0_action.kind == "GONG":
                    apply_gong(state, 0, tile, discarder_idx)
                    print(f"Player 0 (AI) calls gong on {tile}")
                    render_state(state)
                    # AI discards after call
                    ai_discard_for_player0(state)
                    new_tile, new_discarder = state.last_discard
                    return handle_calls_after_discard(state, new_discarder, new_tile)

                elif p0_action.kind == "CHOW":
                    seq = list(p0_action.payload)
                    apply_chow(state, 0, seq, tile, discarder_idx)
                    print(f"Player 0 (AI) calls chow: {' '.join(seq)}")
                    render_state(state)
                    # AI discards after call
                    ai_discard_for_player0(state)
                    new_tile, new_discarder = state.last_discard
                    return handle_calls_after_discard(state, new_discarder, new_tile)

    # --- Then, proceed with AI logic for other players (1,2,3) ---
    
    # 0) Check for winning by calling on the discard for players 1,2,3
    for offset in range(1, n_players):
        idx = (discarder_idx + offset) % n_players
        if idx == 0:
            continue  # AI already had a chance
        player = state.players[idx]

        # Concealed tiles + the candidate winning tile
        test_concealed = player.concealed.copy()
        test_concealed.append(tile)

        if is_winning_hand_with_exposed(test_concealed, player.exposed):
            # AI players always take win
            print(f"\nPlayer {idx} ({player.name}) (AI) wins by taking {tile}!")
            player.concealed.append(tile)
            # Remove tile from discarder's discard pile
            if state.players[discarder_idx].discards and state.players[discarder_idx].discards[-1] == tile:
                state.players[discarder_idx].discards.pop()
            log_action(state, idx, "win", {
                "hand": sort_tiles(test_concealed),
                "winning_tile": tile
            })
            print(f"{player.name} wins with hand:", " ".join(sort_tiles(test_concealed)))
            return -1  # Signal game end

    
    # 1) Pung/Gong priority for players 1,2,3 (automatic AI decisions)
    pung_kong_candidates = []
    for offset in range(1, n_players):
        idx = (discarder_idx + offset) % n_players
        if idx == 0:
            continue  # AI already had a chance
        opts = pung_gong_options(state.players[idx], tile)
        if opts:
            pung_kong_candidates.append((idx, opts))

    chosen_player = None
    chosen_action = None

    if pung_kong_candidates:
        # AI players decide automatically
        for idx, opts in pung_kong_candidates:
            choice = ai_wants_pung_gong(idx, opts)
            if choice is not None:
                chosen_player = idx
                chosen_action = choice
                print(f"\nPlayer {idx} ({state.players[idx].name}) (AI) calls {choice} on {tile}")
                break

    if chosen_player is not None:
        # someone called pung or gong
        if chosen_action == "pung":
            apply_pung(state, chosen_player, tile, discarder_idx)
            render_state(state)
            # caller discards immediately (no draw)
            if chosen_player == 1:
                ai1_discard_random(state, chosen_player)
            elif chosen_player == 2:
                ai2_discard_wind_dragon_then_isolated(state, chosen_player)
            elif chosen_player == 3:
                ai3_discard_most_seen(state, chosen_player)
            # Handle calls on the new discard recursively
            new_tile, new_discarder = state.last_discard
            return handle_calls_after_discard(state, new_discarder, new_tile)
        elif chosen_action == "gong":
            apply_gong(state, chosen_player, tile, discarder_idx)
            render_state(state)
            # after gong + supplement draw, caller must discard
            if chosen_player == 1:
                ai1_discard_random(state, chosen_player)
            elif chosen_player == 2:
                ai2_discard_wind_dragon_then_isolated(state, chosen_player)
            elif chosen_player == 3:
                ai3_discard_most_seen(state, chosen_player)
            # Handle calls on the new discard recursively
            new_tile, new_discarder = state.last_discard
            return handle_calls_after_discard(state, new_discarder, new_tile)

    # 2) Chow only for next player
    chi_player = next_player_idx(discarder_idx)
    if chi_player != 0:
        options = chow_options(state.players[chi_player], tile)
        if options:
            # Player 1 & 2: always chow (take first option)
            # Player 3: never chow
            if chi_player in (1, 2):
                chosen_seq = options[0]
                print(f"\nPlayer {chi_player} ({state.players[chi_player].name}) (AI) calls chow: {' '.join(chosen_seq)}")
                apply_chow(state, chi_player, chosen_seq, tile, discarder_idx)
                render_state(state)
                # caller discards immediately (no draw)
                if chi_player == 1:
                    ai1_discard_random(state, chi_player)
                elif chi_player == 2:
                    ai2_discard_wind_dragon_then_isolated(state, chi_player)
                # Handle calls on the new discard recursively
                new_tile, new_discarder = state.last_discard
                return handle_calls_after_discard(state, new_discarder, new_tile)
            # else: player 3 never chows, pass through

    # 3) No calls; next player in turn
    return next_player_idx(discarder_idx)


# =============================
# TURN EXECUTION
# =============================

def player_discard(state: GameState, player_idx: int):
    """Handle manual discard (unused in current all-AI setup)."""
    player = state.players[player_idx]
    
    # Process any pending pygame events
    for event in pygame.event.get():
        if event.type == pygame.QUIT:
            pygame.quit()
            raise SystemExit
    
    # Update display to show current state (including newly drawn tile)
    render_state(state)
    
    print(f"\n{player.name}'s hand:", " ".join(f"{i}:{t}" for i, t in enumerate(player.concealed)))
    
    while True:
        try:
            inp = input(f"{player.name}, choose index of tile to discard: ").strip()
            idx = int(inp)
            tile = player.concealed.pop(idx)
            player.discards.append(tile)
            state.last_discard = (tile, player_idx)
            log_action(state, player_idx, "discard", {"tile": tile})
            print(f"{player.name} discards {tile}")
            # Update display after discard
            render_state(state)
            return
        except (ValueError, IndexError):
            print("Invalid index, try again.")

def ai_discard_for_player0(state: GameState):
    """
    Use POMCP planner to choose Player 0's discard after making a call.
    Rebuilds belief and plans with current hand configuration.
    """
    global PLAYER0_BELIEF, PLAYER0_PLANNER
    
    # Rebuild belief to reflect the new tiles in hand after the call
    PLAYER0_BELIEF = build_belief_from_public_state(state, num_particles=32)
    
    # Get actions - should be mostly discards at this point
    action = PLAYER0_PLANNER.plan_turn(PLAYER0_BELIEF, state, reacting_to_discard=False)
    
    if action is None or action.kind != "DISCARD":
        # Fallback: greedy discard with state info
        if state.players[0].concealed:
            idx = greedy_discard_index(state.players[0], state)
            tile = state.players[0].concealed.pop(idx)
            state.players[0].discards.append(tile)
            state.last_discard = (tile, 0)
            log_action(state, 0, "discard", {"tile": tile})
            print(f"Player 0 (AI) discards {tile} after call")
        return
    
    # Execute the discard
    idx = action.payload
    tile = state.players[0].concealed.pop(idx)
    state.players[0].discards.append(tile)
    state.last_discard = (tile, 0)
    log_action(state, 0, "discard", {"tile": tile})
    print(f"Player 0 (AI) discards {tile} after call")
    render_state(state)

def player_turn(state: GameState):
    """
    Execute one player's turn: draw tile, check win, make decision (discard or call).
    Player 0 uses POMCP planner; others use heuristic AIs.
    """
    global PLAYER0_BELIEF
    
    pidx = state.current_player
    player = state.players[pidx]
    state.step += 1

    # Draw a tile
    if not state.wall:
        print("Wall is empty. Game ends in draw.")
        return False

    drawn = draw_tile(state, pidx)
    print(f"\n{player.name} draws {drawn}")
    print(f"{player.name}'s hand:", " ".join(player.concealed))
    
    # Update display after drawing
    render_state(state)

    # -------- PLAYER 0: use POMCP-style planner --------
    if pidx == 0:
        # Refresh belief based on latest public info
        PLAYER0_BELIEF = build_belief_from_public_state(state, num_particles=32)
        action = PLAYER0_PLANNER.plan_turn(PLAYER0_BELIEF, state, reacting_to_discard=False)
        if action is None:
            # fallback: random discard
            random_discard_sim(state, 0)
            tile, discarder = state.last_discard
            next_idx = handle_calls_after_discard(state, discarder, tile)
            if next_idx == -1:
                return False
            state.current_player = next_idx
            return True

        kind = action.kind
        if kind == "SELF_WIN":
            tiles_for_win = player.concealed
            log_action(state, pidx, "win", {"hand": sort_tiles(tiles_for_win)})
            print(f"{player.name} (AI) wins with hand:", " ".join(sort_tiles(tiles_for_win)))
            return False

        if kind == "SELF_KONG":
            tile = action.payload
            for _ in range(4):
                player.concealed.remove(tile)
            player.exposed.append([tile]*4)
            draw_tile(state, 0)
            log_action(state, pidx, "self_gong", {"tile": tile})
            print(f"{player.name} (AI) declares concealed gong on {tile}")
            render_state(state)
            # After self-gong, must discard
            # Recursively plan discard
            return player_turn(state)

        if kind == "DISCARD":
            idx = action.payload
            tile = player.concealed.pop(idx)
            player.discards.append(tile)
            state.last_discard = (tile, pidx)
            log_action(state, pidx, "discard", {"tile": tile})
            print(f"{player.name} (AI) discards {tile}")
            render_state(state)

            # Let other players potentially call
            next_idx = handle_calls_after_discard(state, pidx, tile)
            if next_idx == -1:
                return False
            state.current_player = next_idx
            return True

    # -------- Other players: heuristic AIs --------

    # Check self-draw win (tsumo) using concealed + exposed
    if is_winning_hand_with_exposed(player.concealed, player.exposed):
        # Other AIs always take self-draw win
        tiles_for_win = player.concealed
        log_action(state, pidx, "win", {"hand": sort_tiles(tiles_for_win)})
        print(f"{player.name} (AI) wins with hand:", " ".join(sort_tiles(tiles_for_win)))
        return False  # end game

    # Discard according to AI type
    if pidx == 1:
        ai1_discard_random(state, pidx)
    elif pidx == 2:
        ai2_discard_wind_dragon_then_isolated(state, pidx)
    elif pidx == 3:
        ai3_discard_most_seen(state, pidx)
    else:
        # Fallback (should not happen): random discard
        ai1_discard_random(state, pidx)

    render_state(state)
    tile, discarder = state.last_discard

    # Handle calls 
    next_idx = handle_calls_after_discard(state, discarder, tile)
    if next_idx == -1:
        return False  # End game

    state.current_player = next_idx
    return True


# =============================
# MAIN GAME LOOP
# =============================

def deal_initial_hands(state: GameState):
    """Deal 13 tiles to each player."""
    for _ in range(13):
        for i in range(4):
            draw_tile(state, i)

def main():
    """
    Main game loop.
    Initializes POMCP planner for Player 0, runs game until terminal state.
    """
    global PLAYER0_PLANNER, PLAYER0_BELIEF
    
    wall = create_wall()
    players = [Player(name=f"Player {i}") for i in range(4)]
    state = GameState(wall=wall, players=players, current_player=0)

    init_ui()
    deal_initial_hands(state)
    
    # Initialize POMCP planner and belief for Player 0
    print("Initializing AI planner for Player 0...")
    PLAYER0_PLANNER = Player0Planner(num_simulations=1500, max_depth=4, c=1.4, shaping_alpha=0.15)
    PLAYER0_BELIEF = build_belief_from_public_state(state, num_particles=64)
    print(f"AI initialized with {len(PLAYER0_BELIEF.particles)} belief particles, {PLAYER0_PLANNER.num_simulations} sims/turn, depth {PLAYER0_PLANNER.max_depth}.")
    
    print("Initial hands dealt. Starting game.")

    running = True
    clock = pygame.time.Clock()
    while running:
        # Keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                raise SystemExit

        # Run exactly one logical turn
        running = player_turn(state)

        # Render once after the turn's state changes
        render_state(state)
        clock.tick(30)

    print("\nGame over. Final state:")
    render_state(state)
    pygame.display.flip()

    # Keep window open until closed
    waiting = True
    while waiting:
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                waiting = False
        pygame.time.wait(100)


if __name__ == "__main__":
    main()

