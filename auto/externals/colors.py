class Colors:
    DARK_RED = '\033[31m'
    DARK_GREEN = '\033[32m'
    DARK_YELLOW = '\033[33m'
    DARK_BLUE = '\033[34m'
    DARK_MAGENTA = '\033[35m'
    DARK_CYAN = '\033[36m'
    GRAY = '\033[90m'
    LIGHT_RED = '\033[91m'
    LIGHT_GREEN = '\033[92m'
    LIGHT_YELLOW = '\033[93m'
    LIGHT_BLUE = '\033[94m'
    LIGHT_MAGENTA = '\033[95m'
    LIGHT_CYAN = '\033[96m'
    LIGHT_WHITE = '\033[97m'
    RESET = '\033[0m'
    BLACK = '\033[30m'
    RED = '\033[31m'
    GREEN = '\033[32m'
    YELLOW = '\033[33m'
    BLUE = '\033[34m'
    MAGENTA = '\033[35m'
    CYAN = '\033[36m'
    WHITE = '\033[37m'
    BOLD = '\033[1m'
    UNDERLINE = '\033[4m'
    ITALIC = '\033[3m'
    REVERSED = '\033[7m'
    LIGHT_GRAY = '\033[37m'

    ORANGE = '\033[38;5;208m'
    PINK = '\033[38;5;205m'
    PURPLE = '\033[38;5;141m'
    TEAL = '\033[38;5;39m'
    LIME = '\033[38;5;118m'
    DARK_GRAY = '\033[38;5;242m'

def print_color(text, color):
    print(f"{color}{text}{Colors.RESET}")

def format_color(lines: list[str], tags: list[str], colors: list[str]):
    assert len(tags) == len(colors), "Length of tags and colors must be the same"
    ret = []
    for line in lines:
        for tag, color in zip(tags, colors):
            line = line.replace(tag, color)
        ret.append(line)
    return ret
# print_color("This is red text", Colors.RED)
# print_color("This is green text", Colors.GREEN)
# print_color("This is blue text", Colors.BLUE)
# print_color("This is bold text", Colors.BOLD)
# print_color("This is underlined text", Colors.UNDERLINE)
# print_color("This is italic text", Colors.ITALIC)
# print_color("This is reversed text", Colors.REVERSED)
# print_color("This is orange text", Colors.ORANGE)
# print_color("This is pink text", Colors.PINK)
# print_color("This is purple text", Colors.PURPLE)
# print_color("This is teal text", Colors.TEAL)
# print_color("This is lime text", Colors.LIME)
# print_color("This is dark gray text", Colors.DARK_GRAY)
