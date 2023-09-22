import time
import curses
from threading import Thread, Event
from os.path import join

from .loops import inference_step
from .getters import load_lang, get_model

from typing import Any

APP_NAME = "Language Modeling with LSTMs TUI"
APP_VERSION = "v0.1"
REPO = "https://github.com/filippodaniotti/NLU-UniTN-2022"
INSTRUCTIONS = [
    "How to use:",
    "* Type a prompt to generate a text.",
    "* Hit <Enter> to flush the input buffer.",
    "* Hit <Backspace> to delete the character left to the cursor.",
    "* Type 't=' followed by a temperature value to set the temperature.",
    "* Type 'quit' and press <Enter> to exit  the application.",
]

INFO_Y_OFFSET = 7
INSTRUCTIONS_X_OFFSET = 30
SERVICE_WINDOW_Y_OFFSET = 7
INPUT_WINDOW_Y_OFFSET = 5
OUTPUT_WINDOW_Y_OFFSET = 3

def display_info(stdscr, text, y_offset=0, attr=curses.A_NORMAL, left=False):
    height, width = stdscr.getmaxyx()
    y = height // 2 + y_offset
    # centered by default
    x = (width - len(text)) // 2
    if left:
        x = (width // 2) - INSTRUCTIONS_X_OFFSET
    stdscr.addstr(y, x, text, attr)
    stdscr.refresh()

def display_in_subwin(window, text, cursor_x = None, attr=curses.A_NORMAL):
    window.erase()
    window.addstr(0, 0, text, attr)
    if cursor_x is not None:
        window.move(0, cursor_x)
    window.refresh()

def init_stdscr(stdscr):
    curses.curs_set(1)  
    stdscr.clear()      
    stdscr.refresh()    

    display_info(stdscr, f"{APP_NAME}, {APP_VERSION}", -INFO_Y_OFFSET, curses.A_BOLD)
    display_info(stdscr, f"{REPO}", -INFO_Y_OFFSET+2, curses.A_UNDERLINE)
    for idx, line in enumerate(INSTRUCTIONS):
        display_info(stdscr, line, -INFO_Y_OFFSET+5+idx, left=True)


def handle_input(
        key: int, 
        buffer: str, 
        output: str, 
        cursor_x: int, 
        iw: Any, 
        temp: float,
    ) -> tuple[str, str, int, int, bool, bool, float]:
    """
    Handles user input. Specifically, it handles the following events:
    - Enter: 
        * flush the input buffer
        * if the buffer contains the string 'quit', exit the application
        * if the buffer contains the string 't=', set the temperature to the 
            value following the '=' 
    - Backspace: delete the character left to the cursor
    - Arrows: move the cursor left or right
    - Printable characters: append them to the input buffer

    Args:
        key (int): the key pressed by the user
        buffer (str): the input buffer
        output (str): the output buffer
        cursor_x (int): the cursor position in the input buffer
        iw (Any): the input window
        temp (float): the current temperature for the smoothing of the 
            inference output

    Returns:
        str: the updated input buffer
        str: the updated output buffer
        int: the updated cursor position
        bool: whether the user requested to exit the application
        bool: whether the key pressed is a printable character
        float: the updated temperature
    """
    is_printable = False
    exit_requested = False

    # Enter
    if key == curses.KEY_ENTER or key == 10:
        if "quit" in buffer.lower():
            exit_requested = True
        if "t=" in buffer.lower():
            value = buffer.split("=")[1].strip()
            try:
                temp = min(abs(float(value)), 1.0)
            except ValueError:
                value = value.split(" ")[0]
                temp = min(abs(float(value)), 1.0)

        buffer = ""
        output = ""
        cursor_x = 0

    # Backspace
    elif key == curses.KEY_BACKSPACE or key == 127:
        if cursor_x > 0:
            buffer = buffer[:cursor_x - 1] + buffer[cursor_x:]
            cursor_x -= 1

    # Arrows
    elif key == curses.KEY_RIGHT or key == 27:
        next_key = iw.getch()  # Get the next character to identify arrow key
        if next_key == 91:  # Check for escape sequence
            arrow_key = iw.getch()
            if arrow_key == 67:  # Right arrow
                if cursor_x < len(buffer):
                    cursor_x += 1
            elif arrow_key == 68:  # Left arrow
                if cursor_x > 0:
                    cursor_x -= 1

    # printable characters
    elif key >= 32 and key <= 126:
        is_printable = True
        buffer = buffer[:cursor_x] + chr(key) + buffer[cursor_x:]
        cursor_x += 1

    return (buffer, output, cursor_x, 
            exit_requested, is_printable, temp)


def main(config: dict[str, Any], inf_config: dict[str, Any]) -> callable:
    """
    Closure for the main function of the TUI application.

    Args:
        config (dict[str, Any]): the configuration for the model
        inf_config (dict[str, Any]): the configuration for the inference

    Returns:
        callable: the main function of the TUI application
    """
    device = "cpu"
    lang = load_lang(join(*inf_config["lang_path"]))
    model = get_model(config, len(lang), train=False)

    def _main(stdscr):
        
        init_stdscr(stdscr)

        # Subwindows declaration
        service_window = curses.newwin(1, curses.COLS - 2, 
                                    curses.LINES - SERVICE_WINDOW_Y_OFFSET, 1)
        output_window = curses.newwin(2, curses.COLS - 2, 
                                    curses.LINES - OUTPUT_WINDOW_Y_OFFSET, 1)
        input_window = curses.newwin(1, curses.COLS - 2, 
                                    curses.LINES - INPUT_WINDOW_Y_OFFSET, 1)
        input_window.border()  
        input_window.refresh()

        # Helper variables
        temp = [1.0]
        buffer = [""]
        output = [""]
        cursor_x = 0
        is_printable = [False]
        exit_requested = False

        # event for stopping the thread
        stop_thread = Event()

        def _inf_wrapper() -> None:
            """
            Wrapper for the inference step. All the information is shared
            leveraging the mutable lists defined above in the scope of the
            main function. This is a simple workaround to enable
            inter-thread communication.
            """
            while not stop_thread.is_set():
                time.sleep(.5)
                if len(buffer[0]) > 0 and is_printable[0]:
                    output[0] = inference_step(model, buffer[0], lang, inf_config, temp[0], device)
                    display_in_subwin(output_window, output[0], attr=curses.A_ITALIC)
                    input_window.refresh()
                    is_printable[0] = False

        th = Thread(target=_inf_wrapper, args=())
        th.start()

        while True:
            key = input_window.getch()
            buffer[0], output[0], cursor_x, exit_requested, is_printable[0], temp[0] \
                = handle_input(key, buffer[0], output[0], cursor_x, input_window, temp[0])
            
            if exit_requested:
                stop_thread.set()
                for i in range(3, 0, -1):
                    display_in_subwin(
                        service_window, 
                        f"Bye! Application is quitting in {i}...", 
                        0, curses.A_STANDOUT)
                    time.sleep(1)
                th.join()
                break
            
            # Redraw subwindows
            display_in_subwin(output_window, output[0], attr=curses.A_ITALIC)
            display_in_subwin(service_window, f"T: {temp[0]:.2f}")
            display_in_subwin(input_window, buffer[0], cursor_x)

    return _main

def launch_tui(config: dict[str, Any], inf_config: dict[str, Any]) -> None:
    """
    Launches the TUI application.

    Args:
        config (dict[str, Any]): the configuration for the model
        inf_config (dict[str, Any]): the configuration for the inference
    """
    curses.wrapper(main(config, inf_config))
