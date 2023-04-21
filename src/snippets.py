from random import seed

import gradio as gr
import mlflow as ml


def greet(name):
    return "Hello " + name + "!"


def main2():
    demo = gr.Interface(fn=greet, inputs="text", outputs="text")

    demo.launch()


with ml.start_run(description='Outer 1'):
    seed(77)
    with ml.start_run(description='Inner 1', nested=True):
        ...
    with ml.start_run(description='Inner 2', nested=True):
        ...
    with ml.start_run(description='Inner 3', nested=True):
        ...
