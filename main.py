from dotenv import load_dotenv
load_dotenv()

from fastapi import FastAPI
import gradio as gr

from ui.gradio_interface import create_gradio_app

app = FastAPI()

gr_app = create_gradio_app()
app = gr.mount_gradio_app(app, gr_app, path="/")

# 실행: uvicorn main:app --reload