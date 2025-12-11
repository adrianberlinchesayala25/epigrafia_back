import requests
import sys

# 🔴 PON AQUÍ TU URL DE RENDER
RENDER_URL = "https://epigrafia-backend.onrender.com"  # <--- CAMBIA ESTO

def test_backend():
    print(f"📡 Conectando a {RENDER_URL}...")
    
    # 1. Test Health
    try:
        print("\n1️⃣  Probando /api/health...")
        response = requests.get(f"{RENDER_URL}/api/health")
        if response.status_code == 200:
            print("   ✅ Health Check OK:", response.json())
        else:
            print("   ❌ Error en Health:", response.status_code)
    except Exception as e:
        print(f"   ❌ Error de conexión: {e}")
        return

    # 2. Test Analyze (si tienes un audio de prueba)
    # Busca un archivo de audio en la carpeta
    audio_file = "test_audio.wav" # O el nombre de un audio que tengas
    
    import os
    if not os.path.exists(audio_file):
        # Intentar buscar cualquier wav
        wavs = [f for f in os.listdir('.') if f.endswith('.wav')]
        if wavs:
            audio_file = wavs[0]
        else:
            print("\n⚠️ No encuentro archivos .wav para probar el análisis.")
            print("   (Sube un audio a esta carpeta para probar /api/analyze)")
            return

    print(f"\n2️⃣  Probando /api/analyze con '{audio_file}'...")
    try:
        with open(audio_file, 'rb') as f:
            files = {'audio': (audio_file, f, 'audio/wav')}
            response = requests.post(f"{RENDER_URL}/api/analyze", files=files)
            
        if response.status_code == 200:
            print("   ✅ Análisis OK!")
            print("   📊 Respuesta:", response.json())
        else:
            print("   ❌ Error en Análisis:", response.status_code)
            print("   ", response.text)
            
    except Exception as e:
        print(f"   ❌ Error enviando audio: {e}")

if __name__ == "__main__":
    test_backend()
