import requests, time, os
from ascii_magic import AsciiArt, from_image, from_url
from PIL import Image
from io import BytesIO

# Send on Song change
def change_song_glasses(
    song: str,
    author: str, # --> Vibe?
) -> None:

    url = "http://localhost:3000/change-song"
    payload = {
        "song": song,
        "author": author
    }

    response = requests.post(url, data=payload)

    if response.status_code != 200:

        print(f'Failed to send change_song_glasses {response.text}')
    
    return

# Send alert
def send_alert_glasses(
    type: str,
    message: str,
) -> None:
    """Formated on glasses as f'{type}:{message}'
    Args:
        type: str - Type of alert
        message: str - body of alert
    """

    url = "http://localhost:3000/notification"
    payload = {
        "type": type,
        "message": message
    }

    response = requests.post(url, data=payload)

    if response.status_code != 200:

        print(f'Failed to send send_glasses_alert {response.text}')
    
    return

def send_ascii_image(
    image_url: str, # Assumes local host or remote
) -> None:

    aspect_ratio = 2.2

    try:

        if image_url.startswith('http'):

            response = requests.get(image_url)

            img = Image.open(BytesIO(response.content))

            my_art = from_url(image_url)
        
        else:

            img = Image.open(image_url)

            my_art = from_image(image_url)

        width, height = img.size

        columns = aspect_ratio * (5 * width) // height

        t = my_art.to_terminal(columns=columns, width_ratio=aspect_ratio, monochrome=True)

    except Exception as e:

        print(f'Image to ascii failed {e}')

        # Exit early
        return

    url = 'http://localhost:3000/cover-image'

    data = {
            'ascii': t,
    }

    response = requests.post(url, data=data)

    if response.status_code != 200:

        print(f'Failed to send_ascii_art {response.text}')
    
    return

if __name__ == '__main__':

    print('Sending song')

    # Test song
    change_song_glasses(song='Seven Nation Army', author='The White Stripes')

    print('Sleep for 3 seconds')

    time.sleep(3)

    print('Sending Alert')

    send_alert_glasses('ALERT', 'You are tired!')

    print('Sleep for 6 seconds')

    time.sleep(6)

    # Change here
    image_url = '../image.png'

    if os.path.exists(image_url):

        send_ascii_image(image_url)

    else:
        print('Add a image.png or smth')

    print('Sleeping for 3')

    # test remote ascii art
    send_ascii_image('https://cdn1.suno.ai/image_01088ec0-8e8a-4d98-827d-c3359a8aef32.jpeg')