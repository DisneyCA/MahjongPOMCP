#!/usr/bin/env python3
"""
Tournament script to run Mahjong.py 100 times and tally scores.

Scoring rules:
- Winner: +1 point
- Discarder (if someone won by taking the discard): -1 point
- Self-draw (tsumo): winner gets +1 point (no penalty to others)
"""

import sys
import os
from collections import defaultdict

# Suppress pygame support prompt but keep display enabled
os.environ['PYGAME_HIDE_SUPPORT_PROMPT'] = '1'

# Try to install pygame if not available
try:
    import pygame
except ImportError:
    print("pygame not found. Installing...")
    import subprocess
    subprocess.check_call([sys.executable, "-m", "pip", "install", "pygame"])
    import pygame

import importlib.util

# Import Mahjong module
spec = importlib.util.spec_from_file_location("mahjong_game", "Mahjong.py")
mahjong = importlib.util.module_from_spec(spec)
spec.loader.exec_module(mahjong)


def run_single_game():
    """
    Run a single Mahjong game and return the outcome.
    
    Returns:
        dict with keys:
            - 'winner': player index (0-3) who won
            - 'win_type': 'self_draw' or 'discard'
            - 'discarder': player index who discarded winning tile (or None if self-draw)
    """
    # Create new game state
    wall = mahjong.create_wall()
    players = [mahjong.Player(name=f"Player {i}") for i in range(4)]
    state = mahjong.GameState(wall=wall, players=players, current_player=0)
    
    # Initialize UI (with display)
    mahjong.init_ui()
    mahjong.deal_initial_hands(state)
    
    # Initialize POMCP planner for Player 0
    mahjong.PLAYER0_PLANNER = mahjong.Player0Planner(
        num_simulations=1500, 
        max_depth=4, 
        c=1.4, 
        shaping_alpha=0.15
    )
    mahjong.PLAYER0_BELIEF = mahjong.build_belief_from_public_state(state, num_particles=64)
    
    # Run game until terminal state
    running = True
    max_turns = 200  # Safety limit to prevent infinite loops
    turn_count = 0
    clock = pygame.time.Clock()
    
    while running and turn_count < max_turns:
        # Handle pygame events to keep window responsive
        for event in pygame.event.get():
            if event.type == pygame.QUIT:
                pygame.quit()
                return None
        
        running = mahjong.player_turn(state)
        turn_count += 1
        
        # Render the state
        mahjong.render_state(state)
        pygame.display.flip()
        clock.tick(30)  # Limit to 30 FPS
    
    # Show final state briefly
    mahjong.render_state(state)
    pygame.display.flip()
    pygame.time.wait(2000)  # Wait 2 seconds to view final state
    
    # Check game log for winner
    for entry in reversed(state.log):
        if entry['action'] == 'win':
            winner_idx = entry['player']
            info = entry['info']
            
            # Determine if it was self-draw or discard win
            if 'winning_tile' in info:
                # Won by taking a discard
                # Find who discarded the winning tile
                winning_tile = info['winning_tile']
                discarder_idx = None
                
                # Look backwards through log to find the discard of that tile
                for prev_entry in reversed(state.log[:state.log.index(entry)]):
                    if prev_entry['action'] == 'discard' and prev_entry['info'].get('tile') == winning_tile:
                        discarder_idx = prev_entry['player']
                        break
                
                return {
                    'winner': winner_idx,
                    'win_type': 'discard',
                    'discarder': discarder_idx
                }
            else:
                # Self-draw win (tsumo)
                return {
                    'winner': winner_idx,
                    'win_type': 'self_draw',
                    'discarder': None
                }
    
    # Game ended in draw
    return None


def run_tournament(num_games=100):
    """
    Run tournament of multiple games and tally scores.
    """
    scores = defaultdict(int)
    wins_by_player = defaultdict(int)
    self_draw_wins = defaultdict(int)
    discard_wins = defaultdict(int)
    penalties = defaultdict(int)
    draws = 0
    
    print("=" * 60)
    print(f"Running Mahjong Tournament: {num_games} games")
    print("=" * 60)
    print()
    
    for game_num in range(1, num_games + 1):
        print(f"Game {game_num}/{num_games}...", end=" ", flush=True)
        
        outcome = run_single_game()
        
        if outcome is None:
            print("Draw")
            draws += 1
        else:
            winner = outcome['winner']
            win_type = outcome['win_type']
            discarder = outcome['discarder']
            
            # Award point to winner
            scores[winner] += 1
            wins_by_player[winner] += 1
            
            if win_type == 'self_draw':
                print(f"Player {winner} wins (self-draw)")
                self_draw_wins[winner] += 1
            else:
                print(f"Player {winner} wins (discard by Player {discarder})")
                discard_wins[winner] += 1
                # Penalize discarder
                if discarder is not None:
                    scores[discarder] -= 1
                    penalties[discarder] += 1
        
        # Close pygame window between games
        pygame.display.quit()
        pygame.quit()
    
    # Print final results
    print()
    print("=" * 60)
    print("TOURNAMENT RESULTS")
    print("=" * 60)
    print()
    
    print(f"Total games played: {num_games}")
    print(f"Draws: {draws}")
    print()
    
    print("FINAL SCORES:")
    print("-" * 60)
    for player_idx in range(4):
        print(f"Player {player_idx}: {scores[player_idx]:+4d} points")
    print()
    
    print("WIN STATISTICS:")
    print("-" * 60)
    for player_idx in range(4):
        total_wins = wins_by_player[player_idx]
        self_wins = self_draw_wins[player_idx]
        disc_wins = discard_wins[player_idx]
        print(f"Player {player_idx}: {total_wins:3d} wins "
              f"({self_wins} self-draw, {disc_wins} discard)")
    print()
    
    print("PENALTIES (discarded winning tile):")
    print("-" * 60)
    for player_idx in range(4):
        print(f"Player {player_idx}: {penalties[player_idx]:3d} penalties")
    print()
    
    print("=" * 60)
    
    # Determine champion
    max_score = max(scores[i] for i in range(4))
    champions = [i for i in range(4) if scores[i] == max_score]
    
    if len(champions) == 1:
        print(f"🏆 CHAMPION: Player {champions[0]} with {max_score:+d} points!")
    else:
        print(f"🏆 TIE: Players {', '.join(map(str, champions))} with {max_score:+d} points!")
    
    print("=" * 60)
    
    return scores


if __name__ == "__main__":
    # Default to 100 games, but allow command line override
    num_games = 100
    if len(sys.argv) > 1:
        try:
            num_games = int(sys.argv[1])
        except ValueError:
            print(f"Invalid number of games: {sys.argv[1]}")
            print("Usage: python3 run_tournament.py [num_games]")
            sys.exit(1)
    
    run_tournament(num_games)
