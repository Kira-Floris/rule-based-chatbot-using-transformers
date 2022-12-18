from fastapi import FastAPI
from pydantic import BaseModel
import nest_asyncio
from pyngrok import ngrok
import uvicorn

app = FastAPI()

class Text(BaseModel):
  text: str

@app.post('/')
async def predict(text:Text):
  link, title = get_response(str(text))
  return {
      'link':link,
      'title':title
  }

ngrok_tunnel = ngrok.connect(8000)
print('Public URL:', ngrok_tunnel.public_url)
nest_asyncio.apply()
uvicorn.run(app, port=8000)