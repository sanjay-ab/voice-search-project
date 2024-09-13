"""Use to combine parts of phone timing files into a single file."""
import os
from natsort import natsorted

if __name__ == "__main__":
    language = "odia"

    directory = f"data/{language}/analysis/phone_timings"
    output_file = f"data/{language}/analysis/phone_all_mpr.ctm"

    files = natsorted(os.listdir(directory))

    with open(output_file, "w") as outfile:
        for file in files:
            with open(f"{directory}/{file}", "r") as infile:
                for line in infile:
                    outfile.write(f"{language}_{line}")