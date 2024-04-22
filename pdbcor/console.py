from rich.console import Console as RichConsole
from rich.markdown import Markdown
from tqdm import tqdm


class Console(RichConsole):
    def _h(self, text, level):
        """Print header with given level (e.g. `h1` for `level = 1`)."""
        if not isinstance(level, int) or not 1 <= level <= 6:
            raise ValueError(f"Invalid header level: {level}")
        md = f"{'#'*level} {text}"
        self.print(Markdown(md))

    def h1(self, text):
        self._h(text, 1)

    def h2(self, text):
        self._h(text.upper(), 2)

    def h3(self, text):
        self._h(text, 3)

    def h4(self, text):
        self._h(text, 4)

    class tqdm(tqdm):
        pass


console = Console()
