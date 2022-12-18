# !killall ngrok
import nest_asyncio
from pyngrok import ngrok
import os
import sys

from pyngrok import ngrok

ngrok.set_auth_token('2HIrmgip5iGrS7owPKTdHiioqOl_81Qqfm3F5fwZ532nyxw1n')

# Get the dev server port (defaults to 8000 for Uvicorn, can be overridden with `--port`
# when starting the server
port = sys.argv[sys.argv.index("--port") + 1] if "--port" in sys.argv else 8000
print(port)

# Open a ngrok tunnel to the dev server
public_url = ngrok.connect(port).public_url
print(public_url)
nest_asyncio.apply()
# !streamlit run app.py --server.port 8000