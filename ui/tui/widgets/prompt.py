# ui/tui/widgets/prompt.py
"""
The prompt: a multi-line editor with slash-command completion.

Enter sends and Shift+Enter breaks a line. Terminals that do not implement an
enhanced keyboard protocol cannot tell the two apart, so Ctrl+J is bound to
the same newline action as a fallback.
"""

from __future__ import annotations

from textual import events
from textual.app import ComposeResult
from textual.containers import Vertical
from textual.message import Message
from textual.widgets import OptionList, Static, TextArea
from textual.widgets.option_list import Option

SLASH_COMMANDS: tuple[tuple[str, str], ...] = (
    ("/plan", "Investigate and propose a plan before changing anything"),
    ("/build", "Full access to write and shell tools"),
    ("/quit", "End the session"),
    ("/exit", "End the session"),
)

# What a keypress means while the command list is open.
_NAVIGATION = {
    "up": "up",
    "down": "down",
    "enter": "accept",
    "tab": "accept",
    "escape": "dismiss",
}

PROMPT_PLACEHOLDER = "Ask anything"
WORKING_PLACEHOLDER = "Working..."


class PromptInput(TextArea):
    """A TextArea that submits on Enter and yields to the command list."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    class CommandNavigation(Message):
        def __init__(self, action: str) -> None:
            self.action = action
            super().__init__()

    def __init__(self, **kwargs) -> None:
        super().__init__("", soft_wrap=True, show_line_numbers=False, tab_behavior="focus", **kwargs)
        self.commands_open = False

    async def _on_key(self, event: events.Key) -> None:
        if self.commands_open and event.key in _NAVIGATION:
            event.prevent_default()
            event.stop()
            self.post_message(self.CommandNavigation(_NAVIGATION[event.key]))
            return

        if event.key == "enter":
            event.prevent_default()
            event.stop()
            self.post_message(self.Submitted(self.text))
            return

        if event.key in ("shift+enter", "ctrl+j"):
            event.prevent_default()
            event.stop()
            self.insert("\n")
            return

        await super()._on_key(event)


class PromptArea(Vertical):
    """The prompt input, its command list, and the current mode hint."""

    class Submitted(Message):
        def __init__(self, text: str) -> None:
            self.text = text
            super().__init__()

    def compose(self) -> ComposeResult:
        yield Static("", id="prompt-hint", markup=False)
        yield OptionList(id="slash-commands")
        yield PromptInput(id="prompt-input")

    def on_mount(self) -> None:
        self._hide_commands()
        self.hint.display = False

    # --- Parts ---------------------------------------------------------------

    @property
    def input(self) -> PromptInput:
        return self.query_one("#prompt-input", PromptInput)

    @property
    def commands(self) -> OptionList:
        return self.query_one("#slash-commands", OptionList)

    @property
    def hint(self) -> Static:
        return self.query_one("#prompt-hint", Static)

    # --- Modes ---------------------------------------------------------------

    def set_mode(self, mode: str, hint: str = "") -> None:
        """Locks, unlocks, or repurposes the input.

        `locked` while the agent works, `prompt` when it wants the next
        instruction, `reason` when it wants a sentence explaining a refusal.
        """
        prompt_input = self.input
        prompt_input.disabled = mode == "locked"
        prompt_input.placeholder = WORKING_PLACEHOLDER if mode == "locked" else (hint or PROMPT_PLACEHOLDER)

        self.hint.update(hint)
        self.hint.display = bool(hint)

        if mode != "locked":
            prompt_input.focus()

    def focus_input(self) -> None:
        if not self.input.disabled:
            self.input.focus()

    # --- Command list --------------------------------------------------------

    def on_text_area_changed(self, event: TextArea.Changed) -> None:
        self._sync_commands()

    def on_prompt_input_command_navigation(self, event: PromptInput.CommandNavigation) -> None:
        event.stop()
        if event.action == "up":
            self.commands.action_cursor_up()
        elif event.action == "down":
            self.commands.action_cursor_down()
        elif event.action == "dismiss":
            self._hide_commands()
        else:
            self._accept_command()

    def on_prompt_input_submitted(self, event: PromptInput.Submitted) -> None:
        event.stop()
        self.input.text = ""
        self._hide_commands()
        self.post_message(self.Submitted(event.text))

    def on_option_list_option_selected(self, event: OptionList.OptionSelected) -> None:
        event.stop()
        self._accept_command()

    def _sync_commands(self) -> None:
        text = self.input.text

        # The list is a completion aid for the command word itself: once the
        # user has typed past it, they are writing a prompt.
        if not text.startswith("/") or any(character.isspace() for character in text):
            self._hide_commands()
            return

        matches = [item for item in SLASH_COMMANDS if item[0].startswith(text.lower())]
        if not matches:
            self._hide_commands()
            return

        options = self.commands
        options.clear_options()
        options.add_options([Option(f"{name}   {description}", id=name) for name, description in matches])
        options.highlighted = 0
        options.display = True
        self.input.commands_open = True

    def _accept_command(self) -> None:
        options = self.commands
        index = options.highlighted

        if index is not None:
            command = options.get_option_at_index(index).id or ""
            prompt_input = self.input
            prompt_input.text = f"{command} "
            prompt_input.move_cursor(prompt_input.document.end)

        self._hide_commands()

    def _hide_commands(self) -> None:
        self.commands.display = False
        self.input.commands_open = False
