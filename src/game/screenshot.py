from panda3d.core import WindowProperties, PNMImage
import numpy as np
from PIL import Image
import io

def get_game_state_image(base):
    """
    Capture the current game view and return it as a numpy array.
    
    Args:
        base: The ShowBase instance of the game
        
    Returns:
        numpy.ndarray: The current game view as a numpy array in RGB format
    """
    # Get window properties
    win = base.win
    props = WindowProperties()
    props.setSize(win.getXSize(), win.getYSize())
    
    # Create a PNMImage to store the screenshot
    buffer = PNMImage()
    base.graphicsEngine.renderFrame()
    base.graphicsEngine.extractTextureData(win.getScreenshot(), buffer)
    
    # Convert to numpy array
    img_array = np.frombuffer(buffer.getData(), dtype=np.uint8)
    img_array = img_array.reshape((buffer.getYSize(), buffer.getXSize(), 3))
    
    # Convert from BGR to RGB
    img_array = img_array[..., ::-1]
    
    return img_array

def get_compressed_state(base, quality=85):
    """
    Get the current game state as compressed JPEG bytes.
    This is useful for efficient storage of game states.
    
    Args:
        base: The ShowBase instance of the game
        quality: JPEG compression quality (0-100)
        
    Returns:
        bytes: Compressed JPEG data of the current game state
    """
    # Get the image array
    img_array = get_game_state_image(base)
    
    # Convert to PIL Image
    img = Image.fromarray(img_array)
    
    # Compress to JPEG
    img_byte_arr = io.BytesIO()
    img.save(img_byte_arr, format='JPEG', quality=quality)
    img_byte_arr = img_byte_arr.getvalue()
    
    return img_byte_arr 