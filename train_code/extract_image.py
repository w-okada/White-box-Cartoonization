import ffmpeg
import glob
import os
from PIL import Image
import argparse
import MultiProcessUtil

def arg_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--photo_movie_dir",   default = 'dataset/photo_movie',   type = str)
    parser.add_argument("--cartoon_movie_dir", default = 'dataset/cartoon_movie', type = str)     
    parser.add_argument("--start",             default = 100,                     type = int)
    parser.add_argument("--duration",          default = 600,                     type = int)
    parser.add_argument("--interval",          default = 1,                       type = int)
    args = parser.parse_args()    
    return args


def gen_ss_dir_name(movie_dir):
    out_dir = f"{movie_dir}_ss"
    return out_dir

def gen_r_dur_name(movie_dir):
    out_dir = f"{movie_dir}_ss_r"
    return out_dir

def snapshot_worker(param):
    movie_file = param[0]
    out_dir    = param[1]
    base_name  = os.path.basename(movie_file)

    stream = ffmpeg.input(movie_file)
    out_file_t = os.path.join(out_dir, f'{base_name}_%06d.jpg')
    stream = ffmpeg.output(stream, out_file_t, t=args.duration, ss=args.start, r=args.interval ,f='image2')
    ffmpeg.run(stream)

def snapshot(args):
    params = []
    for movie_dir in [args.photo_movie_dir, args.cartoon_movie_dir]:
        movie_files = glob.glob(os.path.join(movie_dir, "*"))
        out_dir = gen_ss_dir_name(movie_dir)
        os.makedirs(out_dir, exist_ok=True)
        params.extend([(f, out_dir) for f in movie_files])

    ress = MultiProcessUtil.manager(params, snapshot_worker, 4)
     
def crop_and_resize_worker(param):
    input_file  = param[0]
    output_file = param[1]
    im = Image.open(input_file)
    w, h = im.size
    resolution = w if w<h else h
    crop = im.crop(
        ((w-resolution)//2,
         (h-resolution)//2,
         (w+resolution)//2,
         (h+resolution)//2)
    )
    resize = crop.resize((256,256))
    resize.save(output_file,quality=100)

def crop_and_resize(args):
    params = []
    for movie_dir in [args.photo_movie_dir, args.cartoon_movie_dir]:
        input_dir = gen_ss_dir_name(movie_dir)
        output_dir = gen_r_dur_name(movie_dir)
        os.makedirs(output_dir, exist_ok=True)
        input_files = glob.glob(os.path.join(input_dir, '*.jpg'))
        params.extend( [(f, os.path.join(output_dir, os.path.basename(f))) for f in input_files] )
    ress = MultiProcessUtil.manager(params, crop_and_resize_worker, 4)

if __name__ == '__main__':
    args = arg_parser()
    snapshot(args)
    crop_and_resize(args)





