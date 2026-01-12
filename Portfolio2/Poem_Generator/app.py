import gradio as gr
import ollama
import pyttsx3
import time
import os
import atexit
import sys # Import sys for error info

# --- Global list to track temp files for cleanup ---
# We'll add all created .wav files here
TEMP_FILES = []

# --- Cleanup Function ---
def cleanup_temp_files():
    """
    Called on script exit to delete all temporary audio files.
    """
    print("\nCleaning up temporary audio files...")
    for f in TEMP_FILES:
        try:
            if os.path.exists(f):
                os.remove(f)
                print(f"Removed: {f}")
        except Exception as e:
            print(f"Error removing file {f}: {e}")

# Register the cleanup function to run when the script exits
atexit.register(cleanup_temp_files)


# --- Main Logic Function ---
def generate_song_and_speech(word1, word2, word3):
    """
    This function is called by Gradio.
    It takes 3 words, generates lyrics, and then generates speech audio.
    """
    
    # 1. Generate Lyrics with Ollama
    print("Generating lyrics...")
    lyric_prompt = f"""
    You are a poet. Your task is to write a short poem in three paragraphs
    inspired by these three words: {word1}, {word2}, {word3}.
    
    Respond with *only* the poem's text, and no other conversational text.
    The song should be the best song ever, too much creative. Should win in America's got Talent
    """
    
    generated_lyrics = ""
    audio_output_path = None # Default to no audio
    
    try:
        response = ollama.chat(
            model='llama3:8b',
            messages=[{'role': 'user', 'content': lyric_prompt}]
        )
        generated_lyrics = response['message']['content']
    except Exception as e:
        print(f"Ollama error: {e}")
        generated_lyrics = f"Error connecting to Ollama: {e}\n(Is the Ollama application running?)"
        return generated_lyrics, None

    # 2. Generate Speech Audio File
    print("Generating speech audio...")
    
    # --- NEW: Create a unique filename for every request ---
    # This prevents the browser/Gradio from caching the old audio file.
    temp_audio_file = f"temp_song_output_{int(time.time())}.wav"
    TEMP_FILES.append(temp_audio_file) # Add to cleanup list

    try:
        tts_engine = pyttsx3.init()
        if tts_engine is None:
            raise Exception("Could not initialize TTS engine.")
            
        tts_engine.save_to_file(generated_lyrics, temp_audio_file)
        tts_engine.runAndWait()

        # Robustly wait for the file to be created and released
        start_time = time.time()
        file_ready = False
        while time.time() - start_time < 5: # 5 second timeout
            if os.path.exists(temp_audio_file) and os.path.getsize(temp_audio_file) > 0:
                try:
                    with open(temp_audio_file, 'r') as f:
                        pass # File exists and is readable
                    file_ready = True
                    break
                except IOError:
                    time.sleep(0.1) # File is locked, wait
            time.sleep(0.1) # File doesn't exist yet, wait
        
        if file_ready:
            print(f"Audio saved to {temp_audio_file}")
            audio_output_path = temp_audio_file
        else:
            print("Timeout or error waiting for audio file.")
            
    except Exception as e:
        print(f"TTS error: {e}")
        pass

    # Return the lyrics (text) and the *unique* path to the audio file
    return generated_lyrics, audio_output_path

# --- Gradio Interface Definition ---
print("Starting Gradio interface...")
app = None # Initialize app to None

try:
    with gr.Blocks(theme=gr.themes.Soft()) as app:
        gr.Markdown(
            """
            # 🎵 3-Word Song Generator
            Enter three words to inspire a song, and the AI will write it.
    
            """
        )
        
        with gr.Row():
            # Inputs
            with gr.Column(scale=1):
                word_1 = gr.Textbox(label="Word 1", placeholder="e.g., Moon")
                word_2 = gr.Textbox(label="Word 2", placeholder="e.g., River")
                word_3 = gr.Textbox(label="Word 3", placeholder="e.g., Echo")
                submit_btn = gr.Button("Generate Song", variant="primary")
                
            # Outputs
            with gr.Column(scale=2):
                output_lyrics = gr.Textbox(label="Generated Lyrics", lines=10)
                output_audio = gr.Audio(
                    label="Spoken Song", 
                    type="filepath",
                    autoplay=True
                )

        # Connect the button click to the function
        submit_btn.click(
            fn=generate_song_and_speech,
            inputs=[word_1, word_2, word_3],
            outputs=[output_lyrics, output_audio]
        )
    print("Gradio interface defined successfully.")

except Exception as e:
    print("\n" + "="*50)
    print("CRITICAL ERROR: Failed to define the Gradio interface.")
    print(f"Error: {e}")
    print(f"Error type: {type(e)}")
    print("This is often due to an issue with the Gradio installation.")
    print("Try running: pip install --upgrade gradio")
    print("="*50 + "\n")
    

# --- Launch the App ---
if __name__ == "__main__":
    if app is not None:
        try:
            print("Ollama app must be running in the background.")
            print("Launching Gradio app... (This will block the terminal)")
            app.launch()
            print("Gio app has closed.") # This will print when you stop the server
        except Exception as e:
            print("\n" + "="*50)
            print("CRITICAL ERROR: Failed to launch the Gradio app.")
            print(f"Error: {e}")
            print(f"Error type: {type(e)}")
            print("This could be a port conflict. Is something else running on port 7860?")
            print("="*50 + "\n")
    else:
        print("Failed to initialize 'app', cannot launch.")