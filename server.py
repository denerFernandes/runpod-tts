# docker/server.py
import os
import uvicorn
import tempfile
import torch
import logging
import time
import gc
import numpy as np
import soundfile as sf
import librosa
from fastapi import FastAPI, UploadFile, File, Form, HTTPException
from fastapi.responses import FileResponse
from fastapi.middleware.cors import CORSMiddleware
from TTS.api import TTS
from contextlib import asynccontextmanager
from pathlib import Path
from TTS.tts.utils.text.cleaners import portuguese_cleaners
from pydub import AudioSegment
import subprocess
import runpod
import base64
from typing import Dict, Any

# Configurar logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Variável global para o modelo
tts_model = None

# ========== FUNÇÕES ESSENCIAIS PARA ÁUDIO LIMPO ==========

def preprocess_reference_audio(input_path: str, output_path: str) -> bool:
    """Pré-processa áudio de referência para 16kHz mono (ideal para XTTS v2)"""
    try:
        audio, orig_sr = librosa.load(input_path, sr=None, mono=False)
        
        if audio.ndim > 1:
            audio = librosa.to_mono(audio)
        
        if orig_sr != 16000:
            audio = librosa.resample(audio, orig_sr=orig_sr, target_sr=16000)
        
        # Normalização suave
        audio = audio / np.max(np.abs(audio)) * 0.8
        
        sf.write(output_path, audio, 16000, subtype='PCM_16')
        logger.info(f"✅ Áudio de referência: {orig_sr}Hz → 16kHz mono")
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no pré-processamento: {e}")
        return False

def postprocess_to_clean_audio(input_path: str, output_path: str, output_format: str = "wav") -> bool:
    """
    Pós-processa para áudio limpo e claro:
    - Converte para 44.1kHz estéreo
    - Normalização final
    - Suporte para WAV ou MP3
    """
    try:
        audio, sr = sf.read(input_path)
        logger.info(f"📊 Áudio original: {sr}Hz")
        
        # Resample para 44.1kHz (qualidade padrão universal)
        if sr != 44100:
            audio = librosa.resample(audio, orig_sr=sr, target_sr=44100)
        
        # Converter para estéreo
        if audio.ndim == 1:
            audio = np.column_stack([audio, audio])
        
        # Normalização final para áudio limpo
        max_val = np.max(np.abs(audio))
        if max_val > 0:
            audio = audio / max_val * 0.85
        
        if output_format.lower() == "mp3":
            # Salvar temporariamente como WAV e converter para MP3
            temp_wav = output_path.replace('.mp3', '_temp.wav')
            sf.write(temp_wav, audio, 44100, subtype='PCM_16')
            
            # Converter para MP3 usando pydub
            try:
                audio_segment = AudioSegment.from_wav(temp_wav)
                audio_segment.export(output_path, format="mp3", bitrate="320k")
                os.unlink(temp_wav)  # Remove arquivo temporário
                logger.info(f"✅ Áudio limpo: 44.1kHz estéreo MP3 (320kbps)")
            except Exception as e:
                logger.warning(f"⚠️ Falha na conversão MP3 com pydub: {e}")
                # Fallback: usar ffmpeg se disponível
                try:
                    subprocess.run([
                        'ffmpeg', '-i', temp_wav, '-codec:a', 'mp3', 
                        '-b:a', '320k', '-y', output_path
                    ], check=True, capture_output=True)
                    os.unlink(temp_wav)
                    logger.info(f"✅ Áudio limpo: 44.1kHz estéreo MP3 (320kbps) via ffmpeg")
                except Exception as ffmpeg_error:
                    logger.error(f"❌ Erro na conversão MP3: {ffmpeg_error}")
                    # Como último recurso, renomeia WAV para MP3 (não recomendado)
                    os.rename(temp_wav, output_path)
                    return False
        else:
            # Salvar como WAV
            sf.write(output_path, audio, 44100, subtype='PCM_16')
            logger.info(f"✅ Áudio limpo: 44.1kHz estéreo WAV")
        
        return True
        
    except Exception as e:
        logger.error(f"❌ Erro no pós-processamento: {e}")
        return False

def cleanup_temp_files(file_paths: list):
    """Remove arquivos temporários"""
    for path in file_paths:
        if path and os.path.exists(path):
            try:
                os.unlink(path)
            except Exception as e:
                logger.warning(f"⚠️ Falha ao remover {path}: {e}")


def clean_text_for_portuguese_tts(text):
    from TTS.tts.utils.text.cleaners import portuguese_cleaners
    import re
    
    # Expandir abreviações
    abbrevs = {"Dr.": "Doutor", "Dra.": "Doutora", "Sr.": "Senhor", "Sra.": "Senhora"}
    for old, new in abbrevs.items():
        text = text.replace(old, new)
    
    # Aplicar cleaner português
    text = portuguese_cleaners(text)
    
    # Números decimais
    text = re.sub(r'(\d+)\.(\d+)', r'\1 vírgula \2', text)
    text = text.replace('.', ';\n')
    text = text.replace('-', ' ')
    
    return text


async def handler(job) -> Dict[str, Any]:
    global tts_model
    
    # Carregar modelo se ainda não foi carregado (lazy loading)
    if tts_model is None:
        logger.info("🔄 Carregando modelo XTTS v2 sob demanda...")
        try:
            tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", 
                           gpu=torch.cuda.is_available())
            logger.info("✅ Modelo XTTS v2 carregado com sucesso!")
        except Exception as e:
            logger.error(f"❌ Erro ao carregar modelo: {e}")
            return {
                "error": True,
                "message": f"Falha ao carregar modelo TTS: {str(e)}",
                "status_code": 503
            }
    
    job_input = job['input']
    
    # Extrair parâmetros do input
    text = job_input.get('text')
    reference_audio_base64 = job_input.get('reference_audio')  # Base64 do áudio de referência
    language = job_input.get('language', 'pt')
    output_format = job_input.get('output_format', 'wav').lower()
    temperature = job_input.get('temperature', 0.75)
    length_penalty = job_input.get('length_penalty', 1.0)
    repetition_penalty = job_input.get('repetition_penalty', 5.0)
    top_k = job_input.get('top_k', 50)
    top_p = job_input.get('top_p', 0.85)
    speed = job_input.get('speed', 1.0)
    enable_text_splitting = job_input.get('enable_text_splitting', True)
    remove_silence = job_input.get('remove_silence', False)
    use_warmup = job_input.get('use_warmup', False)
    improve_start = job_input.get('improve_start', False)
    gpt_cond_len = job_input.get('gpt_cond_len', 15)
    text_cleaner = job_input.get('text_cleaner', True)

    # Validações
    if not tts_model:
        return {
            "error": True,
            "message": "Modelo não carregado",
            "status_code": 503
        }
    
    if not text or not text.strip():
        return {
            "error": True,
            "message": "Texto vazio",
            "status_code": 400
        }
    
    if not reference_audio_base64:
        return {
            "error": True,
            "message": "Áudio de referência não fornecido",
            "status_code": 400
        }
    
    if output_format not in ["wav", "mp3"]:
        return {
            "error": True,
            "message": "Formato deve ser 'wav' ou 'mp3'",
            "status_code": 400
        }
    
    start_time = time.time()
    ref_path = None
    processed_ref_path = None
    output_path = None
    final_path = None

    try:
        logger.info(f"🎵 TTS {output_format.upper()} - Texto: {len(text)} chars, Idioma: {language}")
        logger.info(f"🎛️ Parâmetros - Temp: {temperature}, Speed: {speed}, Top-p: {top_p}")
        
        # 1. Decodificar e salvar áudio de referência
        try:
            reference_audio_bytes = base64.b64decode(reference_audio_base64)
        except Exception as e:
            return {
                "error": True,
                "message": f"Erro ao decodificar áudio de referência: {str(e)}",
                "status_code": 400
            }
        
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as ref_file:
            ref_file.write(reference_audio_bytes)
            ref_path = ref_file.name
        
        # 2. Pré-processar para 16kHz mono (essencial para XTTS v2)
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as processed_ref_file:
            processed_ref_path = processed_ref_file.name
        
        if not preprocess_reference_audio(ref_path, processed_ref_path):
            return {
                "error": True,
                "message": "Falha no pré-processamento do áudio de referência",
                "status_code": 400
            }
        
        # 3. Gerar áudio usando streaming do XTTS v2
        with tempfile.NamedTemporaryFile(suffix=".wav", delete=False) as output_file:
            output_path = output_file.name
        
        logger.info("🎤 Gerando áudio com streaming...")

        # Usar split_sentences=True para textos grandes (streaming interno do XTTS)
        text_to_process = clean_text_for_portuguese_tts(text.strip()) if text_cleaner else text.strip()
        
        tts_model.tts_to_file(
            text=text_to_process,
            speaker_wav=processed_ref_path,
            language=language,
            file_path=output_path,
            temperature=temperature,
            length_penalty=length_penalty,
            repetition_penalty=repetition_penalty,
            top_k=top_k,
            top_p=top_p,
            speed=speed,
            split_sentences=enable_text_splitting
        )
        
        # 4. Pós-processar para áudio limpo no formato escolhido
        file_extension = "mp3" if output_format == "mp3" else "wav"
        with tempfile.NamedTemporaryFile(suffix=f".{file_extension}", delete=False) as final_file:
            final_path = final_file.name
        
        if not postprocess_to_clean_audio(output_path, final_path, output_format):
            # Fallback para áudio original se pós-processamento falhar
            if output_format == "mp3":
                # Se falhou conversão MP3, retorna WAV
                final_path = output_path
                output_format = "wav"
                file_extension = "wav"
                logger.warning("⚠️ Fallback para WAV devido falha na conversão MP3")
            else:
                final_path = output_path
                logger.warning("⚠️ Usando áudio sem pós-processamento")
        
        # Verificar arquivo final
        if not os.path.exists(final_path) or os.path.getsize(final_path) == 0:
            return {
                "error": True,
                "message": "Falha na geração do áudio",
                "status_code": 500
            }
        
        # 5. Converter áudio para base64
        try:
            with open(final_path, 'rb') as audio_file:
                audio_base64 = base64.b64encode(audio_file.read()).decode('utf-8')
        except Exception as e:
            return {
                "error": True,
                "message": f"Erro ao converter áudio para base64: {str(e)}",
                "status_code": 500
            }
        
        # Estatísticas
        generation_time = time.time() - start_time
        file_size = os.path.getsize(final_path)
        
        try:
            if output_format == "wav":
                final_audio, final_sr = sf.read(final_path)
                duration = len(final_audio) / final_sr
            else:
                # Para MP3, usar pydub para obter duração
                audio_segment = AudioSegment.from_mp3(final_path)
                duration = len(audio_segment) / 1000.0  # ms para segundos
                final_sr = audio_segment.frame_rate
            
            logger.info(f"✅ {output_format.upper()} gerado em {generation_time:.2f}s - {duration:.1f}s @ {final_sr}Hz")
        except Exception as e:
            logger.info(f"✅ {output_format.upper()} gerado em {generation_time:.2f}s - {file_size} bytes")
            duration = None
            final_sr = 44100
            logger.warning(f"⚠️ Erro ao obter metadados do áudio: {str(e)}")
        
        # Retornar resposta JSON com áudio em base64
        return {
            "error": False,
            "audio_base64": audio_base64,
            "metadata": {
                "format": output_format.upper(),
                "generation_time": round(generation_time, 2),
                "file_size": file_size,
                "duration": round(duration, 2) if duration else None,
                "sample_rate": final_sr,
                "text_length": len(text),
                "language": language,
                "parameters": {
                    "temperature": temperature,
                    "length_penalty": length_penalty,
                    "repetition_penalty": repetition_penalty,
                    "top_k": top_k,
                    "top_p": top_p,
                    "speed": speed,
                    "enable_text_splitting": enable_text_splitting,
                    "text_cleaner": text_cleaner
                }
            },
            "version": "3.1.0"
        }
        
    except Exception as e:
        logger.error(f"❌ Erro na síntese: {str(e)}")
        return {
            "error": True,
            "message": f"Erro na síntese: {str(e)}",
            "status_code": 500
        }
    
    finally:
        # Limpar arquivos temporários
        cleanup_temp_files([ref_path, processed_ref_path, output_path, final_path])

def load_tts_model():
    global tts_model
    try:
        logger.info("🔄 Carregando modelo XTTS v2...")
        tts_model = TTS(model_name="tts_models/multilingual/multi-dataset/xtts_v2", 
                       gpu=torch.cuda.is_available())
        logger.info("✅ Modelo XTTS v2 carregado com sucesso!")
        return True
    except Exception as e:
        logger.error(f"❌ Erro ao carregar modelo: {e}")
        return False

def init():
    global tts_model
    logger.info("🚀 Inicializando servidor RunPod...")
    
    if not load_tts_model():
        logger.error("❌ Falha crítica: não foi possível carregar o modelo")
        raise Exception("Falha ao carregar modelo TTS")
    
    logger.info("✅ Inicialização completa!")

# Iniciar com função de init
runpod.serverless.start({
    "handler": handler,
    "init": init
})