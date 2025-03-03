import os
import requests

def generate_tattoo_image(description: str) -> bytes:
    """
    Generate a tattoo image based on the provided description using OpenAI's DALL-E API.

    Args:
        description (str): The description of the tattoo to generate.

    Returns:
        bytes: The generated image in bytes.

    Raises:
        Exception: If the image generation fails.
    """
    api_key = os.getenv('OPENAI_API_KEY')
    if not api_key:
        raise Exception("OpenAI API key not found in environment variables.")

    headers = {
        'Authorization': f'Bearer {api_key}',
        'Content-Type': 'application/json',
    }

    data = {
        'prompt': description,
        'n': 1,
        'size': '512x512',
    }

    response = requests.post('https://api.openai.com/v1/images/generations', headers=headers, json=data)

    if response.status_code == 200:
        image_url = response.json()['data'][0]['url']
        # Fetch the image bytes
        image_response = requests.get(image_url)
        if image_response.status_code == 200:
            return image_response.content
        else:
            raise Exception("Failed to fetch generated image.")
    else:
        raise Exception(f"Image generation failed: {response.text}")