from direct.showbase.ShowBase import ShowBase
from direct.task import Task
from panda3d.core import Point3, Vec3, NodePath
from panda3d.core import AmbientLight, DirectionalLight
from panda3d.core import CollisionTraverser, CollisionHandlerQueue
from panda3d.core import CollisionNode, CollisionBox
from panda3d.core import TransformState, CardMaker
from panda3d.core import loadPrcFileData
import sys
import random
import math
from panda3d.core import getModelPath
import os



# Add project root to model path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
getModelPath().appendDirectory(project_root)

class RacingGame(ShowBase):
    def __init__(self):
        super().__init__()
        
        # Disable mouse control of the camera
        self.disableMouse()
        
        # Set up collision detection
        self.cTrav = CollisionTraverser()
        self.collisionHandler = CollisionHandlerQueue()
        
        # Create the game world
        self.setupWorld()
        self.setupLighting()
        self.setupCar()
        self.setupCamera()
        self.createObstacles()
        self.createFinishLine()  # Add finish line
        self.setupControls()
        
        # Game state
        self.gameStartTime = globalClock.getRealTime()
        self.isGameOver = False
        
        # Start the game loop
        self.taskMgr.add(self.update, "update")

    def setupWorld(self):
        # Create a ground plane using CardMaker
        cm = CardMaker('ground')
        cm.setFrame(-25, 25, -2000, 2000)  # Create a 50x4000 unit plane
        self.ground = self.render.attachNewNode(cm.generate())
        self.ground.setPos(0, 0, -0.1)
        self.ground.setHpr(0, -90, 0)  # Rotate to be flat
        self.ground.setColor(0.6, 0.8, 1.0)  # Light blue color

    def setupLighting(self):
        # Ambient light
        alight = AmbientLight('ambient')
        alight.setColor((0.3, 0.3, 0.3, 1))
        alnp = render.attachNewNode(alight)
        render.setLight(alnp)

        # Directional light (sun)
        dlight = DirectionalLight('directional')
        dlight.setColor((0.8, 0.8, 0.8, 1))
        dlnp = render.attachNewNode(dlight)
        dlnp.setHpr(45, -45, 0)
        render.setLight(dlnp)

    def setupCar(self):
        # Load the car model from the existing file
        self.car = loader.loadModel("assets/3d_models/Car/Car.egg")
        self.car.reparentTo(render)
        self.car.setScale(0.3)  # Make the car smaller
        self.car.setPos(0, 0, 0.6)  # Lift higher above ground

        # Car physics properties
        self.carPos = Point3(0, 0, 0.6)  # Match initial position
        self.carHeading = 0
        self.carSpeed = 0
        self.carVelocity = Vec3(0, 0, 0)
        
        # Car parameters
        self.maxSpeed = 30.0
        self.acceleration = 30.0 * 2 # Double the acceleration to make the car more responsive
        self.turnRate = 90.0
        self.friction = 10.0

        # Collision detection for the car - adjust for new scale and height
        collision = CollisionBox(Point3(0, 0, 0.2), 0.3, 0.6, 0.2)  # Adjusted collision box size and center
        cnodePath = self.car.attachNewNode(CollisionNode('car'))
        cnodePath.node().addSolid(collision)
        self.cTrav.addCollider(cnodePath, self.collisionHandler)

    def createObstacles(self):
        self.obstacles = []
        # Keep 200 obstacles to make the game more challenging
        for i in range(200):  
            # Load the cube model
            obstacle = loader.loadModel("assets/3d_models/Cube.egg")
            obstacle.reparentTo(render)
            
            # Scale the cube to desired size
            size = 2.0
            obstacle.setScale(size)
            
            # Position the obstacle
            x = random.uniform(-20, 20)  # Keep within track width, leaving some margin
            y = random.uniform(50, 1900)  # Start placing after the starting area
            
            # Randomly place some obstacles on the negative side of the track too
            if random.random() < 0.5:
                y = -y
            
            # Don't place obstacles too close to the starting position or finish line
            while (abs(x) < 8 and abs(y) < 30) or (abs(x) < 8 and abs(y) > 1900):
                x = random.uniform(-20, 20)
                y = random.uniform(50, 1900)
                if random.random() < 0.5:
                    y = -y
            
            obstacle.setPos(x, y, size)  # Lift by size to sit on ground
            
            # Set random color
            r = random.uniform(0.2, 1.0)
            g = random.uniform(0.2, 1.0)
            b = random.uniform(0.2, 1.0)
            obstacle.setColor(r, g, b, 1)
            
            # Add collision detection
            collision = CollisionBox(Point3(0, 0, 0), size, size, size)
            cnodePath = obstacle.attachNewNode(CollisionNode(f'obstacle{i}'))
            cnodePath.node().addSolid(collision)
            
            self.obstacles.append(obstacle)

    def setupCamera(self):
        # Create a camera rig
        self.cameraRig = NodePath('cameraRig')
        self.cameraRig.reparentTo(render)
        
        # Set up camera parameters
        self.cameraDistance = -10
        self.cameraHeight = 5
        self.cameraAngle = -12
        
        # Position the camera
        self.camera.reparentTo(self.cameraRig)
        self.camera.setPos(0, self.cameraDistance, self.cameraHeight)
        self.camera.setHpr(0, self.cameraAngle, 0)

    def setupControls(self):
        # Driving controls
        self.accept("arrow_up", self.setControl, ["acceleration", 1])
        self.accept("arrow_up-up", self.setControl, ["acceleration", 0])
        self.accept("arrow_down", self.setControl, ["acceleration", -1])
        self.accept("arrow_down-up", self.setControl, ["acceleration", 0])
        self.accept("arrow_left", self.setControl, ["turning", 1])
        self.accept("arrow_left-up", self.setControl, ["turning", 0])
        self.accept("arrow_right", self.setControl, ["turning", -1])
        self.accept("arrow_right-up", self.setControl, ["turning", 0])
        
        # Camera controls
        self.accept("page_up", self.adjustCamera, ["height", 1])
        self.accept("page_down", self.adjustCamera, ["height", -1])
        self.accept("[", self.adjustCamera, ["distance", -1])
        self.accept("]", self.adjustCamera, ["distance", 1])
        
        # Exit control
        self.accept("escape", sys.exit)
        
        # Initialize control states
        self.controls = {
            "acceleration": 0,  # -1 for reverse, 0 for none, 1 for forward
            "turning": 0       # -1 for right, 0 for none, 1 for left
        }

    def setControl(self, control, value):
        self.controls[control] = value

    def adjustCamera(self, setting, value):
        if setting == "height":
            self.cameraHeight = max(2, min(15, self.cameraHeight + value))
        elif setting == "distance":
            self.cameraDistance = max(-20, min(-5, self.cameraDistance - value))
        self.camera.setPos(0, self.cameraDistance, self.cameraHeight)

    def createFinishLine(self):
        # Create a finish line using CardMaker
        self.finishLine = NodePath('finishLine')
        self.finishLine.reparentTo(render)

        # Create the faces of the finish line gate
        cm = CardMaker('finish_line')
        
        # Make it span the width of the track and be tall enough
        width = 25.0  # Half the track width
        height = 5.0
        depth = 0.5
        
        # Front face
        cm.setFrame(-width, width, 0, height)
        front = self.finishLine.attachNewNode(cm.generate())
        front.setY(depth)
        
        # Back face
        back = self.finishLine.attachNewNode(cm.generate())
        back.setY(-depth)
        back.setH(180)
        
        # Left face
        cm.setFrame(-depth, depth, 0, height)
        left = self.finishLine.attachNewNode(cm.generate())
        left.setX(-width)
        left.setH(90)
        
        # Right face
        right = self.finishLine.attachNewNode(cm.generate())
        right.setX(width)
        right.setH(-90)
        
        # Top face
        cm.setFrame(-width, width, -depth, depth)
        top = self.finishLine.attachNewNode(cm.generate())
        top.setZ(height)
        top.setP(-90)
        
        # Position at the end of the track
        self.finishLine.setPos(0, 1950, 0)  # Just before the end of the track
        
        # Color all faces blue
        for child in self.finishLine.getChildren():
            child.setColor(0.2, 0.2, 1.0)  # Blue color
        
        # Add collision detection
        collision = CollisionBox(Point3(0, 0, height/2), width, depth, height/2)
        cnodePath = self.finishLine.attachNewNode(CollisionNode('finishLine'))
        cnodePath.node().addSolid(collision)

    def checkWin(self):
        entries = []
        for i in range(self.collisionHandler.getNumEntries()):
            entry = self.collisionHandler.getEntry(i)
            if entry.getIntoNode().getName() == 'finishLine':
                # Calculate score based on time
                endTime = globalClock.getRealTime()
                timeTaken = endTime - self.gameStartTime
                score = max(1000 - int(timeTaken * 10), 0)  # Deduct points based on time
                print(f"\nCONGRATULATIONS! You've won!")
                print(f"Time: {timeTaken:.2f} seconds")
                print(f"Score: {score} points")
                self.isGameOver = True
                return True
            entries.append(entry)
        return len(entries) > 0

    def update(self, task):
        if self.isGameOver:
            return Task.done
            
        dt = globalClock.getDt()
        
        # Update car physics
        # Handle acceleration
        if self.controls["acceleration"] != 0:
            self.carSpeed += self.controls["acceleration"] * self.acceleration * dt
            self.carSpeed = max(-self.maxSpeed/2, min(self.maxSpeed, self.carSpeed))
        else:
            # Apply friction
            if abs(self.carSpeed) < self.friction * dt:
                self.carSpeed = 0
            else:
                self.carSpeed -= math.copysign(self.friction * dt, self.carSpeed)
        
        # Handle turning
        if self.controls["turning"] != 0:
            turn_amount = self.controls["turning"] * self.turnRate * dt
            # Reduce turning at higher speeds
            turn_amount *= 1.0 - (abs(self.carSpeed) / self.maxSpeed) * 0.5
            self.carHeading += turn_amount
        
        # Update car position
        heading_rad = math.radians(self.carHeading)
        velocity = Vec3(
            -math.sin(heading_rad) * self.carSpeed,
            math.cos(heading_rad) * self.carSpeed,
            0
        )
        self.carPos += velocity * dt
        
        # Check if car is outside the plane boundaries
        if abs(self.carPos.x) > 25 or abs(self.carPos.y) > 2000:
            print("You went off the track! Game Over!")
            self.isGameOver = True
            return Task.done
        
        # Update car visual position and rotation
        self.car.setPos(self.carPos)
        self.car.setH(self.carHeading)
        
        # Update camera position
        target_pos = self.carPos
        current_pos = self.cameraRig.getPos()
        self.cameraRig.setPos(current_pos + (target_pos - current_pos) * min(1.0, dt * 5.0))
        self.cameraRig.setH(self.carHeading)
        
        # Check for collisions
        self.cTrav.traverse(render)
        
        # First check for finish line
        if self.checkWin():
            return Task.done
            
        # Then check for obstacles
        if self.collisionHandler.getNumEntries() > 0:
            print("BOOM! Game Over!")
            self.isGameOver = True
            return Task.done
        
        return Task.cont

    def _get_distance_to_finish(self):
        """Calculate distance to finish line"""
        return abs(2000 - self.carPos.y)

if __name__ == "__main__":
    print(f"Running game from {__file__}")
    # Configure Panda3D window
    loadPrcFileData("", """
        window-title Racing Game
        win-size 1280 720
        framebuffer-multisample 1
        multisamples 2
        show-frame-rate-meter 1
    """)
    # Create and run the game
    game = RacingGame()
    game.run() 