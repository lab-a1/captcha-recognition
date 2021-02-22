import os
import csv


images_path = "../dataset/images"
csv_output_file = "../dataset/ground-truth.csv"
with open(csv_output_file, "w") as file:
    csv_writter = csv.writer(file, delimiter=",")
    csv_writter.writerow(["image_file", "label"])

    images = list(os.listdir(images_path))
    data = [[x, x[0 : len(x) - 4]] for x in images]
    csv_writter.writerows(data)

print("Done.")
