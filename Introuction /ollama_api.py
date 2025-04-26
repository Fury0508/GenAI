from fastapi import FastAPI

app = FastAPI()
client = client(
    host = 'mention api'
)

@app.post("/chat")
def chat():
    pass