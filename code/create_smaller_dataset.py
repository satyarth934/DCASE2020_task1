import os
import sys
import glob
import tqdm
import shutil
import pathlib


def main():
	input_dir = "./data"
	output_dir = "./small_data"

	classes = [d for d in os.listdir(input_dir) if d.startswith("TAU") and (d.split(".")[-1] != "zip")]

	for c in tqdm.tqdm(classes):
		print("class:", c)
		glob_path = f"{input_dir}/{c}/TAU-urban-acoustic-scenes-2020-3class-development/audio/*"
		number_samples = 20
		file_paths = glob.glob(glob_path)[:number_samples]

		for fpath in file_paths:
			opath = fpath.replace(input_dir, output_dir)
			pathlib.Path(os.path.dirname(opath)).mkdir(parents=True, exist_ok=True)
			shutil.copyfile(fpath, opath)


if __name__ == '__main__':
	main()