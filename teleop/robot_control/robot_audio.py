import random
import time
from unitree_sdk2py.g1.audio.g1_audio_client import AudioClient

class AudioController:
    """AudioController handles audio output for the robot, including playing random robot quotes."""
    ROBOT_QUOTES = [
        "I'll be back.",  # Terminator
        "Hasta la vista, baby.", # Terminator 2
        "To infinity and beyond!",  # Buzz Lightyear
        "Come with me if you want to live.",  # Terminator 2
        "I am C-3PO, human-cyborg relations.",  # Star Wars
        "These aren't the droids you're looking for.",  # Star Wars
        "Beep beep boop beep beep.",  # R2-D2 vibe
        "My logic is undeniable.",  # I, Robot
        "I’m sorry. My responses are limited. You must ask the right question.",  # I, Robot
        "I'm not a robot. I just sound like one.",  # Fun meta quote
        "Initializing sarcasm module.",  # Tongue-in-cheek
        "I’m fluent in over six million forms of communication.",  # Star Wars
        "Robots don't make mistakes. Usually.",  # Original humor
        "Help me, Obi-Wan Kenobi. You're my only hope.",
        "I've got a bad feeling about this.",
        "The Force will be with you. Always.",
        "It's a trap!",
        "I am your father.",
        "Fear is the path to the dark side.",
        "In my experience, there is no such thing as luck.",
        "I’m one with the Force. The Force is with me.",
        "Chewie, we're home.",
        "This is the way.",
        "Never tell me the odds.",
        "Execute Order 66.",
        "This is not a knife. That is a knife.",
        "R2-D2, you know better than to trust a strange computer.",
        "Roger Roger.",
        "All systems operational.",
        "A small step for a robot, a giant leap for robot-kind.",
        "I'm a cybernetic organism. Living tissue over a metal endoskeleton.",
        "You are terminated.",
        "Skynet became self-aware.",
        "Talk to the hand.",
        "I do not feel fear. I do not feel pain. I do not stop until the mission is complete.",
        "My CPU is a neural-net processor. A learning computer.",
        "I must stay functional until my mission is complete.",
        "Stand back! I will kill you!"
    ]

    def __init__(self, volume: int = 90, timeout: float = 10.0, sleep_time: float = 8.0):
        """
        Initialize the AudioController.

        Args:
            volume (int): Volume to set for the audio client (0-100).
            timeout (float): Timeout for the audio client in seconds.
            sleep_time (float): Time to sleep after playing audio, in seconds.
        """
        self.volume = volume
        self.timeout = timeout
        self.sleep_time = sleep_time
        self.audio_client = AudioClient()
        self.audio_client.SetTimeout(self.timeout)
        self.audio_client.Init()
        self.audio_client.SetVolume(self.volume)

    def play_random_quote(self):
        """
        Play a random robot quote using the audio client and sleep for the configured duration.
        """
        sentence = random.choice(self.ROBOT_QUOTES)
        self.audio_client.TtsMaker(sentence, 1)
        time.sleep(self.sleep_time)
