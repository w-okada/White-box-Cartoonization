import ffmpeg
import glob
import os
from PIL import Image

scene_movies = ['dataset/photo_movie_scene', 'dataset/cartoon_movie_scene']
face_movies  = ['dataset/photo_movie_face',  'dataset/cartoon_movie_face']

def gen_ss_dir_name(movie_dir):
    out_dir = f"{movie_dir}_ss"
    return out_dir

def gen_r_dur_name(movie_dir):
    out_dir = f"{movie_dir}_ss_r"
    return out_dir


def snapshot():
    for d in scene_movies:
        files = glob.glob(os.path.join(d, "*"))
        print(d)
        out_dir = gen_ss_dir_name(movie_dir)
        os.makedirs(out_dir, exist_ok=True)
        for i, file in enumerate(files):
            stream = ffmpeg.input(file)
            out_file_t = os.path.join(out_dir, f'{i}_%06d.jpg')
            # 開始 100秒後から10分間 (600枚)
            stream = ffmpeg.output(stream, out_file_t, t=600, ss=100, r=1 ,f='image2')
            ffmpeg.run(stream)

    for d in face_movies:
        files = glob.glob(os.path.join(d, "*"))
        print(d)
        out_dir = gen_ss_dir_name(movie_dir)
        os.makedirs(out_dir, exist_ok=True)
        for i, file in enumerate(files):
            stream = ffmpeg.input(file)
            out_file_t = os.path.join(out_dir, f'{i}_%06d.jpg')
            # 開始 100秒後から10分間 (600枚)
            stream = ffmpeg.output(stream, out_file_t, t=600, ss=100, r=1 ,f='image2')
            ffmpeg.run(stream)

def crop_and_resize():
    movie_dirs = scene_movies + face_movies
    print(movie_dirs)
    for d in movie_dirs:
        input_dir = gen_ss_dir_name(d)
        output_dir = gen_r_dur_name(d)
        os.makedirs(output_dir, exist_ok=True)
        input_files = glob.glob(os.path.join(input_dir, '*.jpg'))
        for f in input_files:
            output_file = os.path.join(output_dir, os.path.basename(f))
            print(output_file)
            im = Image.open(f)
            w, h = im.size
            resolution = w if w<h else h
            crop = im.crop(
                ((w-resolution)//2,
                 (h-resolution)//2,
                 (w+resolution)//2,
                 (h+resolution)//2)
            )
            resize = crop.resize((256,256))
            resize.save(output_file,quality=95)


if __name__ == '__main__':
    #snapshot()
    crop_and_resize()





