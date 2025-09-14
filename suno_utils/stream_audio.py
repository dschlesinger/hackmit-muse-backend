import vlc
import time

def stream_audio(url):
    player = vlc.MediaPlayer(url)
    player.play()
    
    # Wait for it to start
    while player.get_state() not in [vlc.State.Playing, vlc.State.Ended]:
        time.sleep(0.1)
    
    # Wait for it to finish
    while player.get_state() != vlc.State.Ended:
        time.sleep(1)