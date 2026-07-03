import os
import re
import json
import cv2
import numpy as np
import mediapipe as mp
import yt_dlp
from youtube_transcript_api import YouTubeTranscriptApi
import google.generativeai as genai

def configurar_gemini(api_key):
    genai.configure(api_key=api_key)

def extrair_video_id(url):
    try:
        ydl_opts = {'quiet': True, 'extract_flat': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(url, download=False)
            if 'id' in info:
                return info['id']
    except Exception as e:
        print(f"Erro ao extrair ID com yt_dlp: {e}")
    
    # Caso falhe, usa uma limpeza manual atualizada para links curtos
    if "youtu.be/" in url:
        # Pega a parte após o youtu.be/ e limpa qualquer interrogação
        parte_id = url.split("youtu.be/")[1]
        video_id = parte_id.split("?")[0].split("&")[0]
        return video_id
        
    padrao = r'(?:v=|\/shorts\/|\/embed\/|\/v\/|\/el\/|watch\?v=)([^#\&\?]*).*'
    match = re.search(padrao, url)
    return match.group(1) if match else None
def obter_transcricao_com_timestamps(url):
    video_id = extrair_video_id(url)
    if not video_id:
        return None
    try:
        transcricao = YouTubeTranscriptApi.get_transcript(video_id, languages=['pt'])
        texto_estruturado = ""
        for trecho in transcricao:
            tempo_inicio = int(trecho['start'])
            minutos = tempo_inicio // 60
            segundos = tempo_inicio % 60
            timestamp = f"[{minutos:02d}:{segundos:02d}]"
            texto_estruturado += f"{timestamp} {trecho['text']}\n"
        return texto_estruturado
    except Exception as e:
        print(f"Erro ao obter transcrição: {e}")
        return None

def analisar_cortes_com_gemini(transcricao_texto):
    model = genai.GenerativeModel('gemini-1.5-pro')
    prompt = f"""
    Você é o especialista em retenção e viralização do canal "Conexão Nerd Cortes". 
    Analise a transcrição do vídeo do Peter Jordan abaixo e selecione exatamente os 7 melhores trechos com maior potencial de viralização para o YouTube Shorts.

    Critérios:
    1. Duração: Máximo de 60 segundos por corte.
    2. Gancho: O início deve conter uma frase impactante ou reação forte.
    3. Conclusão: O trecho precisa fazer sentido sozinho.

    Retorne o resultado ESTREITAMENTE no formato JSON padrão abaixo, sem markdown, sem blocos de código e sem textos adicionais:
    [
      {{"corte": 1, "inicio_segundos": 125, "fim_segundos": 180, "titulo": "Teoria Doutor Destino", "motivo": "Gancho forte"}}
    ]

    Transcrição do vídeo:
    {transcricao_texto}
    """
    resposta = model.generate_content(prompt)
    try:
        texto_limpo = resposta.text.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpo)
    except Exception as e:
        print(f"Erro ao decodificar JSON: {e}")
        return None

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

        historico_x.append(centro_x_detectado)
        if len(historico_x) > tamanho_janela_suavizacao:
            historico_x.pop(0)
        
        centro_x_suave = int(np.mean(historico_x))
        ultimo_centro_x = centro_x_suave

        inicio_x = centro_x_suave - (largura_saida // 2)
        if inicio_x < 0: inicio_x = 0
        elif inicio_x + largura_saida > largura_orig: inicio_x = largura_orig - largura_saida

        frame_recortado = frame[0:altura_saida, inicio_x:inicio_x + largura_saida]
        out.write(frame_recortado)
        atual_frame += 1

    cap.release()
    out.release()
    if os.path.exists("temp_video.mp4"): os.remove("temp_video.mp4")
    return True
