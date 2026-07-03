import streamlit as st
import os
import core_functions as core

st.set_page_config(page_title="Conexão Nerd Cortes", page_icon="🎬")

st.title("⚡ Conexão Nerd Cortes — Cloud")

# Puxa a chave automaticamente do painel do Streamlit Cloud
if "GEMINI_API_KEY" in st.secrets:
    api_key = st.secrets["GEMINI_API_KEY"]
    core.configurar_gemini(api_key)
else:
    st.error("Chave API Key não configurada nos Secrets do Streamlit.")
    st.stop()

url_youtube = st.text_input("🔗 Insira o link do vídeo do YouTube:")

if st.button("🚀 Analisar Vídeo"):
    if not url_youtube:
        st.warning("Insira um link.")
    else:
        with st.spinner("📥 Baixando transcrição..."):
            transcricao = core.obter_transcricao_com_timestamps(url_youtube)
        if transcricao:
            with st.spinner("🧠 Gemini analisando ganchos..."):
                st.session_state['cortes'] = core.analisar_cortes_com_gemini(transcricao)
        else:
            st.error("Legenda não disponível para este vídeo.")

if 'cortes' in st.session_state and st.session_state['cortes']:
    for item in st.session_state['cortes']:
        st.info(f"**Corte #{item['corte']}** — {item['inicio_segundos']}s até {item['fim_segundos']}s")
        st.write(f"📌 **Título:** {item['titulo']}")
        
        nome_final = f"corte_{item['corte']}.mp4"
        if st.button(f"🎬 Gerar Corte #{item['corte']}", key=f"btn_{item['corte']}"):
            with st.spinner("Processando Auto-Framing..."):
                if core.baixar_e_recortar_dinamico(url_youtube, item['inicio_segundos'], item['fim_segundos'], nome_final):
                    with open(nome_final, "rb") as f:
                        st.download_button("📥 Baixar MP4", f, file_name=nome_final, mime="video/mp4")