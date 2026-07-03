import os
import re
import json
import yt_dlp
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

# Função simplificada: Retorna um texto estruturado para testar a interface sem travar
def obter_transcricao_com_timestamps(url):
    try:
        # Apenas valida se o link é interpretável pelo yt-dlp
        ydl_opts = {'quiet': True, 'extract_flat': True}
        with yt_dlp.YoutubeDL(ydl_opts) as ydl:
            ydl.extract_info(url, download=False)
            
        # Retorna uma transcrição base simulada para o Gemini trabalhar na interface
        texto_simulado = (
            "[00:01] Fala galera! Peter aqui do Ei Nerd.\n"
            "[00:15] Hoje vamos falar sobre uma teoria que vai mudar tudo o que você pensa sobre o Doutor Destino.\n"
            "[00:32] Robert Downey Jr está de volta e essa reviravolta é simplesmente genial por causa do multiverso.\n"
            "[00:48] Se você gostou dessa ideia, deixa o seu like e se inscreve no canal!"
        )
        return texto_simulado
    except Exception as e:
        print(f"Erro ao validar link do vídeo: {e}")
        return None

# IA do Gemini gerando os ganchos baseados no texto estruturado
def analisar_cortes_com_gemini(transcricao_texto):
    model = genai.GenerativeModel('gemini-1.5-pro')
    
    prompt = f"""
    Você é o especialista em retenção e viralização do canal "Conexão Nerd Cortes". 
    Analise a transcrição do vídeo do Peter Jordan abaixo e selecione exatamente 2 trechos estratégicos para o YouTube Shorts.

    Critérios cruciais para cada corte:
    1. Duração: Máximo de 60 segundos por corte.
    2. Gancho: O início deve conter uma frase impactante ou reação forte.

    Retorne o resultado ESTREITAMENTE no formato JSON padrão abaixo, sem markdown, sem blocos de código e sem textos adicionais. Apenas o JSON puro:
    [
      {{"corte": 1, "inicio_segundos": 1, "fim_segundos": 32, "titulo": "A volta de RDJ como Doutor Destino", "motivo": "Gancho forte falando sobre o retorno do ator"}},
      {{"corte": 2, "inicio_segundos": 32, "fim_segundos": 60, "titulo": "A reviravolta do Multiverso", "motivo": "Explicação que gera muita retenção e comentários"}}
    ]

    Transcrição do vídeo:
    {transcricao_texto}
    """
    
    try:
        resposta = model.generate_content(prompt)
        texto_limpo = resposta.text.replace("```json", "").replace("```", "").strip()
        return json.loads(texto_limpo)
    except Exception as e:
        print(f"Erro ao decodificar JSON do Gemini: {e}")
        # Retorno de segurança caso a IA falhe na formatação
        return [
            {"corte": 1, "inicio_segundos": 1, "fim_segundos": 32, "titulo": "A volta de RDJ como Doutor Destino", "motivo": "Gancho forte inicial"},
            {"corte": 2, "inicio_segundos": 32, "fim_segundos": 60, "titulo": "A reviravolta do Multiverso", "motivo": "Teoria geek de alta retenção"}
        ]

# Cria um arquivo de saída leve e rápido para simular o download do Short
def baixar_e_recortar_dinamico(url, inicio_seg, fim_seg, nome_saida="corte_vertical.mp4"):
    try:
        # Apenas cria um arquivo de texto disfarçado de vídeo para testar o botão de baixar no Streamlit
        with open(nome_saida, "w") as f:
            f.write("Simulação de arquivo de vídeo gerado com sucesso!")
        return True
    except Exception as e:
        print(f"Erro ao gerar arquivo simulado: {e}")
        return False
