import imageio
import os

def save_animation(env, path_sequence, filename="animation.gif", fps=3):
    frames = []

    for step in path_sequence:
        img = env.render_image(path=[step])
        frames.append(img)

    imageio.mimsave(filename, frames, fps=fps)
    print(f"Saved animation → {filename}")