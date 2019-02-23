#!/usr/bin/env python
"""
A script to measure accuracy against test dataset.

Image base directory contains a set of image directories.
Each image directory contains images belonging to a single category.

__author__ = "Hide Inada"
__copyright__ = "Copyright 2019, Hide Inada"
__license__ = "The MIT License"
__email__ = "hideyuki@gmail.com"
"""
import os
import logging
import sys
import subprocess
import re
from pathlib import Path

log = logging.getLogger(__name__)
logging.basicConfig(level=os.environ.get("LOGLEVEL", "INFO"))  # Change the 2nd arg to INFO to suppress debug logging


def main():
    if len(sys.argv) != 2:
        print("Usage: test.accuracy.py <image base directory name>")
        sys.exit(1)

    image_base_dir = sys.argv[1]
    image_base_dir_path = Path(image_base_dir)
    if image_base_dir_path.exists() is False:
        raise ValueError("Image base directory is not found.")

    with open("accuracy_result.txt", "w") as f_out:

        reg = re.compile("[ |[0-9]|\.")

        score = dict()
        for d in [d for d in image_base_dir_path.iterdir() if d.is_dir()]:
            log.info("Processing " + str(d))

            normalized_directory_name = d.name.lower().replace("_", "")
            print("Dir: " + normalized_directory_name)
            score[normalized_directory_name] = dict()

            score[normalized_directory_name]["total"] = 0
            score[normalized_directory_name]["top5"] = 0
            score[normalized_directory_name]["top1"] = 0

            for f in d.glob("*.jpg"):
                score[normalized_directory_name]["total"] += 1

                log.info("Processing " + str(f))

                output_strings = subprocess.check_output(
                    ["python", "label_image.py", "--graph=/tmp/output_graph.pb", "--labels=/tmp/output_labels.txt",
                     "--input_layer=Placeholder", "--output_layer=final_result", "--image=%s" % (str(f))])

                top5 = output_strings.decode("utf-8")
                top5 = top5.split("\n")
                top5 = list(map(lambda e: reg.sub("", e), top5))[:5]
                # print(top5)

                if normalized_directory_name in top5:
                    log.info("In top5!")
                    score[normalized_directory_name]["top5"] += 1

                if normalized_directory_name == top5[0]:
                    score[normalized_directory_name]["top1"] += 1

            score[normalized_directory_name]["top5_accuracy"] = score[normalized_directory_name]["top5"] / \
                                                                score[normalized_directory_name]["total"]
            score[normalized_directory_name]["top1_accuracy"] = score[normalized_directory_name]["top1"] / \
                                                                score[normalized_directory_name]["total"]

            f_out.write("%s\t%0.4f\t\%0.4f" % (
            normalized_directory_name, score[normalized_directory_name]["top1_accuracy"],
            score[normalized_directory_name]["top5_accuracy"]))


if __name__ == "__main__":
    main()
