import streamlit as st

# Streamlit App
def app():
    st.title("Audio Upload")

    st.write("""
    Upload an audio recording, and it will be displayed here.
    """)

    # File uploader
    uploaded_file = st.file_uploader("Upload Audio", type=["wav", "mp3"])

    if uploaded_file is not None:
        # Display the uploaded audio
        st.audio(uploaded_file, format="audio/wav")
        
        # Optionally display the file name or other details
        st.write(f"File Name: {uploaded_file.name}")

# Run the app
if __name__ == "__main__":
    app()
