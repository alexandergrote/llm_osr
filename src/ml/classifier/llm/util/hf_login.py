from huggingface_hub import login

from src.util.logging import console
from src.util.environment import PydanticEnvironment


def login_to_hf() -> None:

    try:

        env = PydanticEnvironment()

        login(
            token=env.hf_token,
        )

    except ConnectionError:

        console.log("Could not connect to HF due to missing Internet connection")