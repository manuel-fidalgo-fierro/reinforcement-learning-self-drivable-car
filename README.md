# 3D Car Game

A 3D driving game where you control a car in an open world environment while avoiding obstacles. If you crash into an obstacle, your car explodes!

## Requirements

- Python 3.7 or higher
- Panda3D game engine
- NumPy

## Installation

1. Create a virtual environment (recommended):
```bash
python -m venv venv
source venv/bin/activate  # On Windows, use: venv\Scripts\activate
```

2. Install the required packages:
```bash
pip install -r requirements.txt
```

## Running the Game

To start the game, run:
```bash
python src/car_game.py
```

## Controls

- Arrow Up: Accelerate
- Arrow Down: Reverse
- Arrow Left: Turn Left
- Arrow Right: Turn Right
- ESC: Exit game

## Game Features

- 3D open world environment
- Realistic car physics with acceleration and friction
- Multiple obstacles to avoid
- Collision detection
- Third-person camera view
- Explosion effect on collision

## Note

You'll need to provide your own 3D models for:
- The car (`models/car`)
- The environment (`models/environment`)
- The obstacles (`models/box`)

You can find free 3D models on websites like:
- [Kenney Assets](https://kenney.nl/assets)
- [OpenGameArt](https://opengameart.org/)
- [Free3D](https://free3d.com/) 