import json

import cv2
import gradio as gr

from htr_pipeline import read_page, DetectorConfig, LineClusteringConfig, ReaderConfig, PrefixTree

with open('data/words_alpha.txt') as f:
    word_list = [w.strip().upper() for w in f.readlines()]
prefix_tree = PrefixTree(word_list)


def process_page(img, scale, margin, min_words_per_line, use_dictionary):
    read_lines = read_page(img,
                           detector_config=DetectorConfig(scale=scale, margin=margin),
                           line_clustering_config=LineClusteringConfig(min_words_per_line=min_words_per_line),
                           reader_config=ReaderConfig(decoder='word_beam_search' if use_dictionary else 'best_path',
                                                      prefix_tree=prefix_tree))

    res = ''
    for read_line in read_lines:
        res += ' '.join(read_word.text for read_word in read_line) + '\n'

    for i, read_line in enumerate(read_lines):
        for read_word in read_line:
            aabb = read_word.aabb
            cv2.rectangle(img,
                          (aabb.xmin, aabb.ymin),
                          (aabb.xmin + aabb.width, aabb.ymin + aabb.height),
                          (60,185,129),
                          2)

    return res, img


with open('data/config.json') as f:
    config = json.load(f)

demo = gr.Interface(fn=process_page,
             inputs=[gr.Image(label='Uploaded image'),
                     gr.Slider(0, 10, 0.5, step=0.05, label='Box Scale'),
                     gr.Slider(0, 25, 1, step=1, label='Bounded Box Margin'),
                     gr.Slider(1, 10, 1, step=1, label='Minimum no. of words per line'),
                     gr.Checkbox(value=True, label='Use dictionary'),
                     ],
             outputs=[gr.Textbox(label='Read Text'), gr.Image(label='Bounded Text Boxes')],
             allow_flagging='never',
             title='TJPL Handwriting Recognition',
             theme=gr.themes.Ocean(font=[gr.themes.GoogleFont("Inconsolata"), "Arial", "Tahoma"]))
demo.launch()
             