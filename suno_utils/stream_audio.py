import vlc
import time
from dataclasses import dataclass
import threading

class MediaPlayer:
    
    url: str

    def __init__(self, url: str) -> None:

        self.url = url
        self.player = vlc.MediaPlayer(url)

    def play(self) -> None:

        self.player_thread = threading.Thread(target=lambda: self.player.play())

        self.player.play()

        # Wait until playing
        while self.player.get_state() not in [vlc.State.Playing, vlc.State.Ended]:
            time.sleep(0.1)

    def time_left(self) -> float:

        return self.player.duration() * (1 - self.player.get_position())

    def is_playing(self) -> bool:

        return self.player.get_state() == vlc.State.Playing