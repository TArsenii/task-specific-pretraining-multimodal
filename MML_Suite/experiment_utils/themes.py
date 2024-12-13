from rich.theme import Theme

monokai_theme = Theme(
    {
        "info_prefix": "bold cyan",
        "warning_prefix": "bold yellow",
        "error_prefix": "bold red",
        "success_prefix": "bold green",
        "default": "white",
        "highlight": "bold magenta",
        "dim": "dim white",
        "heading": "bright_cyan",
        "note": "italic dim cyan",
        "primary": "#f8f8f2",  # Default text color
        "keyword": "#f92672",  # Pink for keywords
        "builtin": "#66d9ef",  # Light blue for built-in names
        "string": "#e6db74",  # Yellow for strings
        "number": "#ae81ff",  # Purple for numbers
        "operator": "#f92672",  # Pink for operators
        "comment": "dim #75715e",  # Grey for comments
        "error_highlight": "bold #f92672",
    }
)


# Modern and clean theme with high contrast
nord_theme = Theme(
    {
        "info_prefix": "#88C0D0",  # Frost blue
        "warning_prefix": "#EBCB8B",  # Yellow
        "error_prefix": "#BF616A",  # Red
        "success_prefix": "#A3BE8C",  # Green
        "default": "#ECEFF4",  # Snow Storm
        "highlight": "#B48EAD",  # Purple
        "dim": "#4C566A",  # Dark gray
        "heading": "#81A1C1",  # Lighter blue
        "note": "#5E81AC",  # Dark blue
        "primary": "#D8DEE9",  # Light gray
        "keyword": "#81A1C1",  # Blue
        "builtin": "#88C0D0",  # Light blue
        "string": "#A3BE8C",  # Green
        "number": "#B48EAD",  # Purple
        "operator": "#81A1C1",  # Blue
        "comment": "#4C566A",  # Dark gray
        "error_highlight": "#BF616A",  # Red
        "table_dim": "#369e3f",
        "table_bright": "#ECEFF4",
    }
)

# Solarized dark theme
solarized_dark = Theme(
    {
        "info_prefix": "#2AA198",  # Cyan
        "warning_prefix": "#B58900",  # Yellow
        "error_prefix": "#DC322F",  # Red
        "success_prefix": "#859900",  # Green
        "default": "#93A1A1",  # Base1
        "highlight": "#6C71C4",  # Violet
        "dim": "#586E75",  # Base01
        "heading": "#268BD2",  # Blue
        "note": "#2AA198",  # Cyan
        "primary": "#EEE8D5",  # Base2
        "keyword": "#CB4B16",  # Orange
        "builtin": "#2AA198",  # Cyan
        "string": "#859900",  # Green
        "number": "#D33682",  # Magenta
        "operator": "#CB4B16",  # Orange
        "comment": "#586E75",  # Base01
        "error_highlight": "#DC322F",  # Red
    }
)

# Dracula theme
dracula_theme = Theme(
    {
        "info_prefix": "#8BE9FD",  # Cyan
        "warning_prefix": "#FFB86C",  # Orange
        "error_prefix": "#FF5555",  # Red
        "success_prefix": "#50FA7B",  # Green
        "default": "#F8F8F2",  # Foreground
        "highlight": "#FF79C6",  # Pink
        "dim": "#6272A4",  # Comment
        "heading": "#BD93F9",  # Purple
        "note": "#6272A4",  # Comment
        "primary": "#F8F8F2",  # Foreground
        "keyword": "#FF79C6",  # Pink
        "builtin": "#8BE9FD",  # Cyan
        "string": "#F1FA8C",  # Yellow
        "number": "#BD93F9",  # Purple
        "operator": "#FF79C6",  # Pink
        "comment": "#6272A4",  # Comment
        "error_highlight": "#FF5555",  # Red
    }
)

# GitHub theme (light)
github_light = Theme(
    {
        "info_prefix": "#0366D6",  # Blue
        "warning_prefix": "#D15704",  # Orange
        "error_prefix": "#D73A49",  # Red
        "success_prefix": "#28A745",  # Green
        "default": "#24292E",  # Black
        "highlight": "#6F42C1",  # Purple
        "dim": "#6A737D",  # Gray
        "heading": "#005CC5",  # Dark blue
        "note": "#6A737D",  # Gray
        "primary": "#24292E",  # Black
        "keyword": "#D73A49",  # Red
        "builtin": "#005CC5",  # Dark blue
        "string": "#032F62",  # Navy
        "number": "#005CC5",  # Dark blue
        "operator": "#D73A49",  # Red
        "comment": "#6A737D",  # Gray
        "error_highlight": "#CB2431",  # Bright red
    }
)

# One Dark theme
one_dark = Theme(
    {
        "info_prefix": "#56B6C2",  # Cyan
        "warning_prefix": "#E5C07B",  # Yellow
        "error_prefix": "#E06C75",  # Red
        "success_prefix": "#98C379",  # Green
        "default": "#ABB2BF",  # Foreground
        "highlight": "#C678DD",  # Purple
        "dim": "#5C6370",  # Comment gray
        "heading": "#61AFEF",  # Blue
        "note": "#5C6370",  # Comment gray
        "primary": "#ABB2BF",  # Foreground
        "keyword": "#C678DD",  # Purple
        "builtin": "#56B6C2",  # Cyan
        "string": "#98C379",  # Green
        "number": "#D19A66",  # Orange
        "operator": "#56B6C2",  # Cyan
        "comment": "#5C6370",  # Comment gray
        "error_highlight": "#BE5046",  # Dark red
    }
)

# Tokyo Night theme
tokyo_night = Theme(
    {
        "info_prefix": "#7DCFFF",  # Light blue
        "warning_prefix": "#FF9E64",  # Orange
        "error_prefix": "#F7768E",  # Red
        "success_prefix": "#9ECE6A",  # Green
        "default": "#A9B1D6",  # Foreground
        "highlight": "#BB9AF7",  # Purple
        "dim": "#565F89",  # Comment
        "heading": "#7AA2F7",  # Blue
        "note": "#565F89",  # Comment
        "primary": "#C0CAF5",  # Light foreground
        "keyword": "#9D7CD8",  # Purple
        "builtin": "#7DCFFF",  # Light blue
        "string": "#9ECE6A",  # Green
        "number": "#FF9E64",  # Orange
        "operator": "#89DDFF",  # Cyan
        "comment": "#565F89",  # Comment
        "error_highlight": "#F7768E",  # Red
    }
)

# Gruvbox theme
gruvbox_dark = Theme(
    {
        "info_prefix": "#83A598",  # Blue
        "warning_prefix": "#FABD2F",  # Yellow
        "error_prefix": "#FB4934",  # Red
        "success_prefix": "#B8BB26",  # Green
        "default": "#EBDBB2",  # Light beige
        "highlight": "#D3869B",  # Purple
        "dim": "#928374",  # Gray
        "heading": "#83A598",  # Blue
        "note": "#928374",  # Gray
        "primary": "#EBDBB2",  # Light beige
        "keyword": "#FB4934",  # Red
        "builtin": "#83A598",  # Blue
        "string": "#B8BB26",  # Green
        "number": "#D3869B",  # Purple
        "operator": "#FB4934",  # Red
        "comment": "#928374",  # Gray
        "error_highlight": "#CC241D",  # Dark red
    }
)

# Catppuccin theme
catppuccin = Theme(
    {
        "info_prefix": "#89DCEB",  # Sky
        "warning_prefix": "#FAE3B0",  # Yellow
        "error_prefix": "#F28FAD",  # Red
        "success_prefix": "#ABE9B3",  # Green
        "default": "#D9E0EE",  # Text
        "highlight": "#DDB6F2",  # Mauve
        "dim": "#6E6C7E",  # Surface2
        "heading": "#96CDFB",  # Blue
        "note": "#6E6C7E",  # Surface2
        "primary": "#D9E0EE",  # Text
        "keyword": "#F28FAD",  # Red
        "builtin": "#89DCEB",  # Sky
        "string": "#ABE9B3",  # Green
        "number": "#F8BD96",  # Peach
        "operator": "#DDB6F2",  # Mauve
        "comment": "#6E6C7E",  # Surface2
        "error_highlight": "#E8A2AF",  # Maroon
    }
)
THEME = nord_theme
WIDTH_SCALE = 1.0
