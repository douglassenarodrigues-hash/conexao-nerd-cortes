import os
import re
import json
import cv2
import numpy as np
import mediapipe as mp
import yt_dlp
import whisper
import google.generativeai as genai

# Configuração da API do Gemini
def configurar_gemini(api_key):
    genai.configure(api_key=api_key)

# Extrai a ID do vídeo limpando qualquer lixo de rastreamento do link
def extrair_video_id(url):
    if "youtu.be/" in url:
        return url.split("youtu.be/")[1].split("?")[0].split("&")[0]
    padrao = r'(?:v=|\/shorts\/|\/embed\/|\/v\/|\/el\/|watch\?v=)([^#\&\?]*).*'
    match = re.search(padrao, url)
    return match.group(1) if match else "video_generic"

# NOVA FUNÇÃO: Baixa o áudio e usa a IA Whisper para transcrever tudo de forma independente
def obter_transcricao_com_timestamps(url):
    audio_temp = "temp_audio.mp3"
    
    # Configuração do yt-dlp para baixar apenas o áudio o mais rápido possível
    ydl_opts = {
        'format': 'bestaudio/best',
        'outtmpl': 'temp_audio',
        'quiet': True
    }
    
    try:
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.download([url])
        
        # Garante a extensão correta caso o formato mude (webm, m4a, etc.)
        if os.path.exists("temp_audio"): os.rename("temp_audio", audio_temp)
        elif os.path.exists("temp_audio.webm"): os.rename("temp_audio.webm", audio_temp)
        elif os.path.exists("temp_audio.m4a"): os.rename("temp_audio.m4a", audio_temp)

        if not os.path.exists(audio_temp):
            return None

        # Carrega o modelo Whisper do pacote open-ai (versão 'base' é ideal para o servidor)
        model = whisper.load_model("base")
        resultado = model.transcribe(audio_temp, language="pt")

        # Estrutura a fala com os minutos e segundos para o Gemini analisar
        texto_estruturado = ""
        for segmento in resultado['segments']:
            tempo_inicio = int(segmento['start'])
            minutos = tempo_inicio // 60
            segundos = tempo_inicio % 60
            timestamp = f"[{minutos:02d}:{segundos:02d}]"
            texto_estruturado += f"{timestamp} {segmento['text']}\n"

        # Limpa o arquivo de áudio temporário para economizar espaço no servidor
        if os.path.exists(audio_temp): 
            os.remove(audio_temp)
        
        return texto_estruturado

    except Exception as e:
        print(f"Erro na transcrição por áudio: {e}")
        if os.path.exists(audio_temp): 
            os.remove(audio_temp)
        return None

# IA do Gemini analisando onde estão os picos de retenção para criar os 7 Shorts
def analisar_cortes_com_gemini(transcricao_texto):
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = f"""
    Você é o especialista em retenção e viralização do canal "Conexão Nerd Cortes". 
    Analise a transcrição do vídeo do Peter Jordan abaixo (gerada por áudio) e selecione exatamente os 7 melhores trechos com maior potencial de viralização para o YouTube Shorts.

    Critérios cruciais para cada corte:
    1. Duração: Máximo de 60 segundos por corte.
    2. Gancho: O início deve conter uma frase muito impactante, uma pergunta intrigante ou uma reação forte.
    3. Conclusão: O trecho precisa fazer sentido sozinho e terminar em um ponto alto.

    Retorne o resultado ESTREITAMENTE no formato JSON padrão abaixo, sem markdown, sem blocos de código e sem textos adicionais. Apenas a lista JSON limpa:
    [
      {{"corte": 1, "inicio_segundos": 125, "fim_segundos": 180, "titulo": "A polêmica teoria sobre o Doutor Destino", "motivo": "Gancho forte falando sobre o retorno de RDJ"}}
    ]

    Transcrição do vídeo:
    {transcricao_texto}
    """
    
    resposta = model.generate_content(prompt)
    try:
        texto_limpo = resposta.text.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpo)
    except Exception as e:
        print(f"Erro ao decodificar JSON do Gemini: {e}")
        return None

# Motor de processamento de imagem 9:16 com a IA que segue o rosto (MediaPipe + OpenCV)
def baixar_e_recortar_dinamico(url, inicio_seg, fim_seg, nome_saida="corte_vertical.mp4"):
    ydl_opts = {
        'format': 'bestvideo[ext=mp4]+bestaudio[ext=m4a]/best[ext=mp4]/best',
        'outtmpl': 'temp_video.mp4',
        'quiet': True
    }
    
    with yt_dlp.YoutubeDL(ydl_opts) as ydl:
        ydl.download([url])
        
    if not os.path.exists("temp_video.mp4"):
        return False

    # Inicializa o detector facial do Google
    mp_face_detection = mp.solutions.face_detection
    face_detection = mp_face_detection.FaceDetection(model_selection=1, min_detection_confidence=0.4)

    cap = cv2.VideoCapture("temp_video.mp4")
    fps = cap.get(cv2.CAP_PROP_FPS)
    largura_orig = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    altura_orig = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    cap.set(cv2.CAP_PROP_POS_FRAMES, int(inicio_seg * fps))
    frame_final = int(fim_seg * fps)

    altura_saida = altura_orig
    largura_saida = int(altura_saida * (9 / 16))
    
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(nome_saida, fourcc, fps, (largura_saida, altura_saida))

    # Variáveis da Média Móvel (Smoothing) para deixar o movimento da câmera super suave
    historico_x = []
    tamanho_janela_suavizacao = int(fps * 0.8) 
    ultimo_centro_x = largura_orig // 2

    atual_frame = int(inicio_seg * fps)
    while cap.isOpened() and atual_frame <= frame_final:
        sucesso, frame = cap.read()
        if not sucesso:
            break

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        resultados = face_detection.process(frame_rgb)
        centro_x_detectado = None

        if resultados.detections:
            for detection in resultados.detections:
                bbox = detection.location_data.relative_bounding_box
                centro_x_detectado = int((bbox.xmin + bbox.width / 2) * largura_orig)
                break 

        if centro_x_detectado is None:
            centro_x_detectado = ultimo_centro_x

        # Aplica a suavização fluida de movimento (efeito Palco Central do WhatsApp)
        historico_x.append(centro_x_detectado)
        if len(historico_x) > tamanho_janela_suavizacao:
            historico_x.pop(0)
        
        centro_x_suave = int(np.mean(historico_x))
        ultimo_centro_x = centro_x_suave

        inicio_x = centro_x_suave - (largura_saida // 2)
        if inicio_x < 0: 
            inicio_x = 0
        elif inicio_x + largura_saida > largura_orig: 
            inicio_x = largura_orig - largura_saida

        # Faz o crop mantendo o foco dinâmico no rosto
        frame_recortado = frame[0:altura_saida, inicio_x:inicio_x + largura_saida]
        out.write(frame_recortado)
        atual_frame += 1

    cap.release()
    out.release()
    
    if os.path.exists("temp_video.mp4"): 
        os.remove("temp_video.mp4")
        
    return True
