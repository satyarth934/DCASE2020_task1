import os
import sys
import glob
import tqdm
import shutil
import pathlib
import numpy as np


def main():
	input_dir = "./data"
	number_samples = 60
	output_dir = f"./small_data_{number_samples}each"

	classes = [d for d in os.listdir(input_dir) if d.startswith("TAU") and (d.split(".")[-1] != "zip")]

	for c in tqdm.tqdm(classes):
		print("class:", c)
		glob_path = f"{input_dir}/{c}/TAU-urban-acoustic-scenes-2020-3class-development/audio/*"
		# number_samples = 20
		# file_paths = glob.glob(glob_path)[:number_samples]

		rand_idx = np.random.randint(0, len(glob_path), number_samples)
		file_paths_all = glob.glob(glob_path)
		try:
			file_paths = [file_paths_all[ri] for ri in rand_idx]
		except Exception as e:
			continue

		for fpath in file_paths:
			opath = fpath.replace(input_dir, output_dir)
			pathlib.Path(os.path.dirname(opath)).mkdir(parents=True, exist_ok=True)
			shutil.copyfile(fpath, opath)


if __name__ == '__main__':
	main()