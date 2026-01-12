import ollama
import sys

def generate_song_lyrics(word1, word2, word3):
    """
    Generates song lyrics using Ollama based on three inspirational words.
    """
    
    # This is the "prompt" we send to the model.
    # We're telling it *exactly* what we want.
    prompt = f"""
    You are a professional and creative songwriter.
    Your task is to write a complete song with a clear structure.
    
    The song must be inspired by these three words:
    1. {word1}
    2. {word2}
    3. {word3}
    
    Please structure the song with:
    - Verse 1
    - Chorus
    - Verse 2
    - Chorus
    - Bridge
    - Chorus
    
    Generate the lyrics now.
    """

    print(f"Generating song from: {word1}, {word2}, {word3}...\n")
    print("--------------------------------------------------")

    try:
        # Call the Ollama API
        response = ollama.chat(
            model='llama3:8b',  # This is the model we downloaded
            messages=[
                {'role': 'user', 'content': prompt}
            ]
        )
        
        # Return just the text content of the response
        return response['message']['content']

    except Exception as e:
        return f"An error occurred: {e}\n(Is the Ollama application running?)"

if __name__ == "__main__":
    # Check if the user provided exactly 3 words
    if len(sys.argv) != 4:
        print("\nUsage: python song_writer.py <word1> <word2> <word3>")
        print("Example: python song_writer.py ocean whisper stars")
        sys.exit(1)
        
    # Get the three words from the command line arguments
    w1, w2, w3 = sys.argv[1], sys.argv[2], sys.argv[3]
    
    # Generate and print the song
    song = generate_song_lyrics(w1, w2, w3)
    print(song)