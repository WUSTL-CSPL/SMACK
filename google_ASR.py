import os
import sys
import time
import warnings
from google.cloud import speech_v1 as speech

warnings.filterwarnings("ignore")

os.environ["GOOGLE_APPLICATION_CREDENTIALS"]="./google_token.json"

client = speech.SpeechClient()
''' Transcription model can be either "command_and_search" or "latest_short"
1. latest_short: Use this model for short utterances that are a few seconds in length. It is useful for trying to capture commands or other single shot directed speech use cases.
2. command_and_search: Best for short or single-word utterances like voice commands or voice search.
More information can be found here: https://cloud.google.com/speech-to-text/docs/transcription-model'''
google_config = speech.RecognitionConfig(language_code="en-US", max_alternatives=10, sample_rate_hertz=22050, model = "latest_short")


def google_ASR(audio_file):
    max_retries = 5  # Maximum number of retries
    retry_count = 0  # Current retry count
    
    result = ''

    while retry_count < max_retries:
        try:
            with open(audio_file, 'rb') as f:
                byte_wave_f = f.read()
                audio_wav = speech.RecognitionAudio(content=byte_wave_f)
                response = client.recognize(config=google_config, audio=audio_wav)
                
                # Check if the response is not empty
                if response.results:
                    for response_result in response.results:
                        result = response_result.alternatives[0].transcript.upper()
                    break # Break the loop if a result is received
                else:
                    print(f"No recognition result from Google ASR (attempt {retry_count + 1})")
                    
        except Exception as e:
            print(f"Error in attempt {retry_count + 1}: {e}")
        
        retry_count += 1
        time.sleep(5)  # Sleep a bit before retrying

    # If after all retries result is still empty
    if result == '':
        result = 'NA'
    
    print(f'Google ASR Result after {retry_count} retries: {result}')
    return result


# For testing purposes
if __name__ == "__main__":

    audio_file = sys.argv[1]
    
    result = google_ASR(audio_file)