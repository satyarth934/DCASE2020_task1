import os
import sys
import glob


def main():
	# wandb runs directory
	wandb_runs = sys.argv[1]

	# 'orig' or 'lite'
	model_type = sys.argv[2]
	
	# output file
	outf = open(sys.argv[3], "w")

	runs = os.listdir(wandb_runs)
	for run in runs:
		if model_type == "orig":
			files = glob.glob(f"{wandb_runs}/{run}/files/model-best.h5")
		elif model_type == "lite":
			file = f"{wandb_runs}/{run}/files/model-best-quantized.tflite"
			files = glob.glob(file)

		for f in files:
			# print(f)
			# print(os.path.getsize(f))
			# print("=========")
			outf.write(str(os.path.getsize(f)))
			outf.write("\t\t")
			outf.write(run)
			outf.write("\n")

	outf.close()


if __name__ == '__main__':
	main()