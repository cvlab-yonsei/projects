import requests
import tarfile
import os

#taken from https://stackoverflow.com/questions/25010369/wget-curl-large-file-from-google-drive/39225039#39225039
def download_file_from_google_drive(id, destination):
    URL = "https://docs.google.com/uc?export=download"
    session = requests.Session()

    response = session.get(URL, params = { 'id' : id }, stream = True)
    token = get_confirm_token(response)

    if token:
        params = { 'id' : id, 'confirm' : token }
        response = session.get(URL, params = params, stream = True)

    save_response_content(response, destination)    

def get_confirm_token(response):
    for key, value in response.cookies.items():
        if key.startswith('download_warning'):
            return value

    return None

def save_response_content(response, destination):
    CHUNK_SIZE = 32768

    with open(destination, "wb") as f:
        for chunk in response.iter_content(CHUNK_SIZE):
            if chunk: # filter out keep-alive new chunks
                f.write(chunk)

if __name__ == "__main__":
    if not os.path.exists('./data/training_data/'):
        print("Downloading files at './data/training_data/'")
        download_file_from_google_drive('1oP0_zMUksaLe3nZFE-fKBcjLw5rA-PYf', './data/training_data.tar.gz')
        tar = tarfile.open('./data/training_data.tar.gz', "r:gz")
        print('Extracting...')
        tar.extractall('./data/')
        tar.close()
        os.remove('./data/training_data.tar.gz')

    if not os.path.exists('./data/PF_Pascal/'):
        print("Downloading files at './data/PF_Pascal/'")
        download_file_from_google_drive('1YXS1Q7zw8_GV93Vl4UPexndqsJ7La9LQ', './data/PF_Pascal.tar.gz')
        tar = tarfile.open('./data/PF_Pascal.tar.gz', "r:gz")
        print('Extracting...')
        tar.extractall('./data/')
        tar.close()
        os.remove('./data/PF_Pascal.tar.gz')

    if not os.path.exists('./data/PF_WILLOW/'):
        print("Downloading files at './data/PF_WILLOW/'")
        download_file_from_google_drive('1vhXW6BpymkvHtTwqE5hVzU00zutrT_PC', './data/PF_WILLOW.tar.gz')
        tar = tarfile.open('./data/PF_WILLOW.tar.gz', "r:gz")
        print('Extracting...')
        tar.extractall('./data/')
        tar.close()
        os.remove('./data/PF_WILLOW.tar.gz')
    print("Done!")