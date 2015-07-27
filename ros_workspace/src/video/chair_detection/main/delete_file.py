import glob
import os
i=0
x_paths = []
datasets_dir = "/".join(os.path.abspath(__file__).split("/")[:-3])+"/data_storage/chair_data/data_set/data_test/"
# while (i<10):
for d in glob.glob(datasets_dir+"*/"):
	paths = glob.glob(d+"*-full.png")
	x_paths.extend(paths)

for p in x_paths:
	i+=1
	print p
	os.system("rm {0}".format(p))
	if i > 500:
		break
		print "up to 10"
		

