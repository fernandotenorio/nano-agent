import json
from pathlib import Path
from typedefs import Message, SystemMessage, UserMessage, AssistantMessage

class Transcript:
    """
    Manages in-memory message state and disk syncing.
    File format (.jsonl): 1 line = 1 complete serialized Message model.
    """
    def __init__(self, file_path: Path):
        self.file_path = file_path
        self.messages: list[Message] = []
        self.load()

    def load(self) -> None:
        if not self.file_path.exists():
            return
            
        with open(self.file_path, "r", encoding="utf-8") as f:
            for line in f:
                if not line.strip():
                    continue
                data = json.loads(line)
                role = data.get("role")
                
                # Explicit routing based on the 'role' discriminator
                if role == "system":
                    self.messages.append(SystemMessage.model_validate(data))
                elif role == "user":
                    self.messages.append(UserMessage.model_validate(data))
                elif role == "assistant":
                    self.messages.append(AssistantMessage.model_validate(data))

    def append(self, message: Message) -> None:
        self.messages.append(message)
        # Ensure parent dirs exist
        self.file_path.parent.mkdir(parents=True, exist_ok=True)
        # Atomic append to disk
        with open(self.file_path, "a", encoding="utf-8") as f:
            f.write(message.model_dump_json() + "\n")