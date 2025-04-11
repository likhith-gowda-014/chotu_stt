function playTTS(text) {
    fetch("/tts_audio", {
      method: "POST",
      headers: { "Content-Type": "application/json" },
      body: JSON.stringify({ text: text })
    })
    .then(response => response.blob())
    .then(audioBlob => {
      const audioUrl = URL.createObjectURL(audioBlob);
      if (currentAudio) currentAudio.pause();
  
      currentAudio = new Audio(audioUrl);
  
      // 👇 Add talking class when audio starts
      currentAudio.onplay = () => {
        document.getElementById("avatar-container").classList.add("talking");
      };
  
      // 👇 Remove it when done
      currentAudio.onended = () => {
        document.getElementById("avatar-container").classList.remove("talking");
      };
  
      currentAudio.play();
      pauseBtn.disabled = false;
      pauseBtn.innerHTML = '<i class="fas fa-pause"></i>';
    })
    .catch(error => console.error("TTS Error:", error));
  }
  