import curses
from os.path import join

from .loops import train, evaluate, inference
from .getters import load_config, load_lang, get_device, get_model

from typing import Any

APP_NAME = "Language Modeling with LSTMs TUI"
APP_VERSION = "v1.0"
AUTHOR = "Filippo Daniotti"
REPO = "https://github.com/filippodaniotti/NLU-UniTN-2022"


def display_centered_text(stdscr, text, y_offset=0, attr=curses.A_NORMAL):
    height, width = stdscr.getmaxyx()
    stdscr.addstr(height // 2 + y_offset, (width - len(text)) // 2, text, attr)
    stdscr.refresh()

def display_in_subwin(window, text, cursor_x = None):
    window.clear()
    window.addstr(0, 0, text)
    if cursor_x is not None:
        window.move(0, cursor_x)
    window.refresh()

def handle_input(key, buffer, cursor_position, iw):
    exit_requested = False

    # Enter
    if key == curses.KEY_ENTER or key == 10:
        if "quit" in buffer.lower():
            exit_requested = True
        buffer = ""
        cursor_position = 0
    
    # Backspace
    elif key == curses.KEY_BACKSPACE or key == 127:
        if cursor_position > 0:
            buffer = buffer[:cursor_position - 1] + buffer[cursor_position:]
            cursor_position -= 1

    # Arrows
    elif key == curses.KEY_RIGHT or key == 27:
        next_key = iw.getch()  # Get the next character to identify arrow key
        if next_key == 91:  # Check for escape sequence
            arrow_key = iw.getch()
            if arrow_key == 67:  # Right arrow
                if cursor_position < len(buffer):
                    cursor_position += 1
            elif arrow_key == 68:  # Left arrow
                if cursor_position > 0:
                    cursor_position -= 1

    # printable characters
    elif key >= 32 and key <= 126:
        buffer = buffer[:cursor_position] + chr(key) + buffer[cursor_position:]
        cursor_position += 1

    return buffer, cursor_position, exit_requested

def main(config: dict[str, Any], inf_config: dict[str, Any]) -> callable:

    lang = load_lang(join(*inf_config["lang_path"]))
    model = get_model(config, len(lang), train=False)

    def _main(stdscr):
        
        curses.curs_set(1)  
        stdscr.clear()      
        stdscr.refresh()    

        display_centered_text(stdscr, f"{APP_NAME}, {APP_VERSION}", -3, curses.A_BOLD)
        # display_centered_text(stdscr, f"Created by {AUTHOR}", -1)
        display_centered_text(stdscr, f"{REPO}", -1, curses.A_UNDERLINE)

        # Input window
        input_window = curses.newwin(1, curses.COLS - 2, curses.LINES - 5, 1)
        input_window.border()  # Draw a border around the input window
        input_window.refresh()

        # Output window
        output_window = curses.newwin(2, curses.COLS - 2, curses.LINES - 3, 1)

        buffer = ""
        cursor_position = 0
        exit_requested = False

        while True:
            key = input_window.getch()
            buffer, cursor_position, exit_requested = handle_input(key, buffer, cursor_position, input_window)
            if exit_requested: 
                break

            output = model.generate(
                buffer, 
                lang, 
                mode=inf_config["mode"], 
                max_len=inf_config["max_length"],
                allow_unk=inf_config["allow_unk"],
                temperature=1.0,
                device=get_device())
            
            display_in_subwin(output_window, output)
            display_in_subwin(input_window, buffer, cursor_position)

    return _main

def launch_tui(config: dict[str, Any], inf_config: dict[str, Any]):
    curses.wrapper(main(config, inf_config))

if __name__ == "__main__":
    config = load_config("configs/merity_ad_nohh_1024.yaml")
    inf_config = load_config("configs/inference.yaml")
    launch_tui(config, inf_config)
