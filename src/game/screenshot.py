import numpy as np
from panda3d.core import PNMImage, GraphicsOutput, GraphicsPipe, WindowProperties
from PIL import Image
import io

def get_game_state_image(base):
    """
    Capture the current game view and return it as a numpy array.
    
    Args:
        base: The ShowBase instance of the game
        
    Returns:
        numpy.ndarray: The captured image as a numpy array
    """
    # Get the window properties
    win = base.win
    props = WindowProperties()
    props.setSize(win.getXSize(), win.getYSize())
    
    # Create a buffer to store the screenshot
    buffer = PNMImage()
    win.getScreenshot(buffer)
    
    # Create numpy array from the image data
    width = buffer.getXSize()
    height = buffer.getYSize()
    
    # Initialize a numpy array to store the image data
    image = np.zeros((height, width, 3), dtype=np.uint8)
    
    # Copy pixel data manually
    for y in range(height):
        for x in range(width):
            r = int(buffer.getBlue(x, y) * 255)  # Panda3D uses BGR
            g = int(buffer.getGreen(x, y) * 255)
            b = int(buffer.getRed(x, y) * 255)
            image[y, x] = [r, g, b]
    
    return image

def get_compressed_state(base, quality=85):
    """
    Get the current game state as compressed JPEG bytes.
    
    Args:
        base: The ShowBase instance of the game
        quality: JPEG compression quality (0-100)
        
    Returns:
        bytes: Compressed JPEG data
    """
    # Get the image as numpy array
    image = get_game_state_image(base)
    
    # Convert to PIL Image
    pil_image = Image.fromarray(image)
    
    # Compress to JPEG
    output = io.BytesIO()
    pil_image.save(output, format='JPEG', quality=quality)
    return output.getvalue() 