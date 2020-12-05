import os
import pandas as pd


class3_dict = {"indoor": ["airport", "metro_station", "shopping_mall"], 
				"outdoor": ["street_pedestrian", "park", "street_traffic", "public_square"], 
				"transportation": ["metro", "bus", "tram"]
				}


def get_class(file_name):
	fineclass = os.path.basename(file_name).split("-")[0]

	for k in class3_dict:
		if fineclass in class3_dict[k]:
			return k


def main():
	data_path = '../task1b/data_2020/'
	test_csv = data_path + 'evaluation_setup/fold1_test_nolabel.csv'

	test_df = pd.read_csv(test_csv)
	test_df["scene_label"] = test_df.applymap(get_class)
	
	test_df.to_csv(test_csv.replace("fold1_test_nolabel.csv","fold1_test.csv"), sep="\t", index=False)


if __name__ == '__main__':
	main()