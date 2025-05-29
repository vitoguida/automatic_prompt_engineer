import openai

# Configurazione Azure
openai.api_type = "azure"
openai.api_base = "https://dipalma3.openai.azure.com/"
openai.api_version = "2024-05-01-preview"  # Assicurati che sia compatibile col tuo deployment
openai.api_key = ""  # Sostituisci con la tua chiave reale

# Nome del deployment che hai creato su Azure (non il modello, ma il nome che gli hai dato tu)
deployment_name = "gpt-4o-mini"  # Adatta a quello che hai realmente
#deployment_name = "gpt-4-turbo-2024-04-09"

# Messaggi di input stile Chat (ruoli: system, user, assistant)
messages = [
    {"role": "system", "content": "Sei un assistente utile."},
    {"role": "user", "content": "mi dici come si chiama il figlio di bombardiro crocodilo?"}
]

# Chiamata all’API
"""response = openai.ChatCompletion.create(
    engine=deployment_name,
    messages=messages,
    temperature=0.7,
    max_tokens=100,
)
print(response["choices"][0]["message"]["content"])
"""
"""response = openai.Completion.create (
           model="gpt-4-turbo-2024-04-09",
           prompt="red carpet is blue?",
           max_tokens=100,
           logprobs=5,
           api_version="2024-06-01"
       )
print(response)"""

response = openai.ChatCompletion.create(
    engine=deployment_name,
    messages=messages,
    temperature=0.7,
    max_tokens=100,
    logprobs=True,  # <--- deve essere booleano, non un numero
)
print(response["choices"][0]["logprobs"])


