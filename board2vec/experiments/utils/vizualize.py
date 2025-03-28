import streamlit as st
import plotly.express as px
import pandas as pd
import chess
import numpy as np
from sklearn.manifold import TSNE


# Снижение размерности с помощью t-SNE
def reduce_dimensions(embeddings):
    tsne = TSNE(n_components=2, random_state=42, perplexity=30)
    reduced = tsne.fit_transform(embeddings)
    return reduced


# Основная функция
def main(boards: list[chess.Board], embeddings: np.ndarray):
    st.title("Визуализация Board2Vec эмбеддингов")

    # Снижение размерности
    reduced_embeddings = reduce_dimensions(embeddings)

    # Создание DataFrame для Plotly
    import pandas as pd
    df = pd.DataFrame(reduced_embeddings, columns=["x", "y"])
    df["board"] = [str(board).replace("\n", "<br>") for board in boards]

    # Создание интерактивного графика
    fig = px.scatter(
        df,
        x="x",
        y="y",
        title="2D Визуализация Board2Vec эмбеддингов",
        hover_data=["board"]
    )
    fig.update_traces(
        textposition="top center",
        hovertext=df["board"],
        hovertemplate="<span style='font-family: monospace;'>%{hovertext}</span><extra></extra>"
    )
    fig.update_layout(showlegend=False)

    # Отображение графика в Streamlit
    st.plotly_chart(fig, use_container_width=True)