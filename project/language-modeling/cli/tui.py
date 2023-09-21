import time
import curses
from threading import Thread
from os.path import join

from .loops import inference_step
from .getters import load_config, load_lang, get_device, get_model

from typing import Any

APP_NAME = "Language Modeling with LSTMs TUI"
APP_VERSION = "v1.0"
AUTHOR = "Filippo Daniotti"
REPO = "https://github.com/filippodaniotti/NLU-UniTN-2022"

DELAY = 2


def display_centered_text(stdscr, text, y_offset=0, attr=curses.A_NORMAL):
    height, width = stdscr.getmaxyx()
    stdscr.addstr(height // 2 + y_offset, (width - len(text)) // 2, text, attr)
    stdscr.refresh()

def display_in_subwin(window, text, cursor_x = None, attr=curses.A_NORMAL):
    window.erase()
    window.addstr(0, 0, text, attr)
    if cursor_x is not None:
        window.move(0, cursor_x)
    window.refresh()

def handle_input(key, buffer, output, counter, cursor_position, iw, temp):
    exit_requested = False

    # Enter
    if key == curses.KEY_ENTER or key == 10:
        if "quit" in buffer.lower():
            exit_requested = True
        if "t=" in buffer.lower():
            value = buffer.split("=")[1]
            try:
                temp = float(value)
            except ValueError:
                value = value.split(" ")[0]
                temp = float(value)

        buffer = ""
        output = ""
        counter = 0
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
        counter += 1

    return buffer, output, counter, cursor_position, exit_requested, temp


def main(config: dict[str, Any], inf_config: dict[str, Any]) -> callable:

    device = get_device()
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

        # Temperature window
        temp_window = curses.newwin(1, curses.COLS - 2, curses.LINES - 7, 1)

        # Output window
        output_window = curses.newwin(2, curses.COLS - 2, curses.LINES - 3, 1)

        counter = 0
        buffer = ""
        output = [""]
        cursor_position = 0
        exit_requested = False
        temp = 1.0


        def inference_wrapper(buffer, temp, output):
            output[0] = inference_step(model, buffer, lang, inf_config, temp, device)
            display_in_subwin(output_window, output[0], attr=curses.A_ITALIC)
            input_window.refresh()
            return

        while True:
            key = input_window.getch()
            buffer, output[0], counter, cursor_position, exit_requested, temp = handle_input(
                key, buffer, output[0], counter, cursor_position, input_window, temp)
            if exit_requested:
                for i in range(3, 0, -1):
                    display_in_subwin(output_window, f"Application is quittingin {i}...")
                    time.sleep(1)
                break
            
            if len(buffer) > 0 and (counter % DELAY == 0 or buffer[-1] == " "):
                th = Thread(target=inference_wrapper, args=(buffer, temp, output))
                th.start()
                counter = DELAY
            elif counter < DELAY:
                output[0] = buffer
            
            display_in_subwin(output_window, output[0], attr=curses.A_ITALIC)
            display_in_subwin(temp_window, f"T: {temp:.2f}")
            display_in_subwin(input_window, buffer, cursor_position)

    return _main

def launch_tui(config: dict[str, Any], inf_config: dict[str, Any]):
    curses.wrapper(main(config, inf_config))

if __name__ == "__main__":
    config = load_config("configs/merity_ad_nohh_1024.yaml")
    inf_config = load_config("configs/inference.yaml")
    launch_tui(config, inf_config)
