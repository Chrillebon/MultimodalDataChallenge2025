import csv
import re
from collections import defaultdict
import random

# Data to be written to the CSV file
data = []

# Specify the output file name
output_file = "shoppinglist.csv"
metadata_file = "/zhome/c8/5/147202/summerschool25/MultimodalDataChallenge2025/data/metadata/metadata.csv"

if __name__ == "__main__":

    random.seed(42)
    filtered_classes = defaultdict(list)

    numbuyfromeach = 10
    estimated_price = 0
    bought_all = []

    pattern = re.compile(r"fungi_train\d{6}\.jpg")

    with open(metadata_file, mode="r", newline="") as csv_file:
        reader = csv.reader(csv_file)
        for row in reader:
            if row and pattern.fullmatch(row[0]):
                # Assuming class is the last column
                thisclass = int(row[-1])
                filtered_classes[thisclass].append(row[0])

    for thisclass in sorted(list(filtered_classes.keys())):

        addthis = random.sample(
            filtered_classes[thisclass],
            min(numbuyfromeach, len(filtered_classes[thisclass])),
        )
        assert (
            len(set(addthis)) == numbuyfromeach
        ), f"failed for class {thisclass}, where we did not get {numbuyfromeach}, but {len(addthis)}"
        for elem in addthis:
            bought_all.append(elem)
            data.extend(
                [
                    [elem, "Habitat"],
                    [elem, "Latitude"],
                    [elem, "Longitude"],
                    [elem, "Substrate"],
                    [elem, "eventDate"],
                ]
            )
            estimated_price = estimated_price + 8
        # print(
        #     f"for class {thisclass}, we have {len(filtered_classes[thisclass])} elements!"
        # )
    print(
        f"DONE! The estimated price is {estimated_price} for {len(bought_all)} instances"
    )
    # filtered_rows now contains all rows with fungi_trainXXXXXX.jpg and class == target_class

    # Write data to the CSV file
    with open(output_file, mode="w", newline="") as csv_file:
        writer = csv.writer(csv_file)
        # Write rows into the file
        writer.writerows(data)

    print(f"Data has been successfully written to '{output_file}'.")
