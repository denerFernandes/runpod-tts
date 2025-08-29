# RunPod TTS com Armazenamento S3 Tebi

Este serviço gera áudio a partir de texto usando o modelo XTTS v2 e armazena o resultado em um bucket S3 da Tebi.

## Configuração

Para que o serviço funcione corretamente, é necessário configurar as seguintes variáveis de ambiente:

```
TEBI_ACCESS_KEY=sua_access_key
TEBI_SECRET_KEY=sua_secret_key
TEBI_BUCKET_NAME=nome_do_seu_bucket
```

## Uso

O serviço espera receber um JSON com os seguintes campos:

```json
{
  "input": {
    "text": "Texto a ser convertido em áudio",
    "script_id": "identificador_do_script",
    "audio_id": "identificador_do_audio",
    "reference_audio_url": "https://exemplo.com/audio_referencia.wav",
    "language": "pt",
    "output_format": "mp3",
    "temperature": 0.75,
    "speed": 1.0
  }
}
```

### Parâmetros obrigatórios

- `text`: Texto a ser convertido em áudio
- `script_id`: Identificador do script (usado para organizar os arquivos no S3)
- `audio_id`: Identificador do áudio (usado para nomear o arquivo no S3)
- Um dos seguintes parâmetros para o áudio de referência:
  - `reference_audio`: Áudio de referência em base64 (usado para clonar a voz)
  - `reference_audio_url`: URL do áudio de referência (usado para clonar a voz)

### Parâmetros opcionais

- `language`: Idioma do texto (padrão: "pt")
- `output_format`: Formato de saída ("wav" ou "mp3", padrão: "wav")
- `temperature`: Temperatura para geração (padrão: 0.75)
- `length_penalty`: Penalidade de comprimento (padrão: 1.0)
- `repetition_penalty`: Penalidade de repetição (padrão: 2.0)
- `top_k`: Valor de top-k (padrão: 50)
- `top_p`: Valor de top-p (padrão: 0.85)
- `speed`: Velocidade da fala (padrão: 1.0)
- `enable_text_splitting`: Habilitar divisão de texto (padrão: true)
- `remove_silence`: Remover silêncio (padrão: false)
- `use_warmup`: Usar warmup (padrão: false)
- `improve_start`: Melhorar início (padrão: false)
- `gpt_cond_len`: Comprimento de condicionamento GPT (padrão: 15)
- `text_cleaner`: Usar limpador de texto (padrão: true)

## Resposta

O serviço retorna um JSON com os seguintes campos:

```json
{
  "error": false,
  "audio_url": "s3.bucket/script_id/audio/audio_id.mp3",
  "metadata": {
    "format": "MP3",
    "generation_time": 2.5,
    "file_size": 123456,
    "duration": 5.2,
    "sample_rate": 44100,
    "text_length": 100,
    "language": "pt",
    "parameters": {
      "temperature": 0.75,
      "length_penalty": 1.0,
      "repetition_penalty": 2.0,
      "top_k": 50,
      "top_p": 0.85,
      "speed": 1.0,
      "enable_text_splitting": true,
      "text_cleaner": true
    }
  },
  "version": "3.1.0"
}
```

## Estrutura de armazenamento no S3

Os arquivos de áudio são armazenados no S3 com a seguinte estrutura:

```
s3.bucket/{script_id}/audio/{audio_id}.{extensão}
```

Onde:
- `script_id`: Identificador do script fornecido na requisição
- `audio_id`: Identificador do áudio fornecido na requisição
- `extensão`: "mp3" ou "wav" dependendo do formato solicitado