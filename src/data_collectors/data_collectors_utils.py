def window_position(x_pos, y_pos, win_size_w, win_size_h, screen_w, screen_h):
    """
    Calculate the window position for the next instance in a grid layout.
    args:
        x_pos: The current x position of the window.
        y_pos: The current y position of the window.
        win_size_w: The width of the window.
        win_size_h: The height of the window.
        screen_w: The width of the screen.
        screen_h: The height of the screen.
    returns:
        x_pos: The next x position of the window.
        y_pos: The next y position of the window.
    """
    # Calculate how many windows can fit in each row and column
    windows_per_row = screen_w // win_size_w
    windows_per_col = screen_h // win_size_h
    
    # Calculate current position in grid
    current_col = x_pos // win_size_w
    current_row = y_pos // win_size_h
    
    # Move to next position
    current_col += 1
    if current_col >= windows_per_row:
        current_col = 0
        current_row += 1
        if current_row >= windows_per_col:
            current_row = 0
    
    # Calculate new position
    x_pos = current_col * win_size_w
    y_pos = current_row * win_size_h
    
    print(f"Window position: {x_pos}, {y_pos} (Grid position: {current_col}, {current_row})")
    return x_pos, y_pos