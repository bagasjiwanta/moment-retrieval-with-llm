import requests
from tqdm import tqdm
from torch import tensor


def download_large_file(url, download_path):
    '''
    Download large file with tqdm progress bar

    Args:
        url (str): url like.
        save_path (str): path like including file name and extension.
    '''
    try:
        with requests.get(url, stream=True) as response:
            response.raise_for_status()
            total_size = int(response.headers.get('content-length', 0))
            chunk_size = 10 * 1024
            with tqdm(total=total_size, unit='B', unit_scale=True, desc=f'Downloading {download_path.split("/")[-1]}') as progress_bar:
                with open(download_path, 'wb') as f:
                    for chunk in response.iter_content(chunk_size=chunk_size):
                        progress_bar.update(len(chunk))
                        f.write(chunk)
        print("File downloaded successfully!")
    except requests.exceptions.RequestException as e:
        print("Error downloading the file:", e)

