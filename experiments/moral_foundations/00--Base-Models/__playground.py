import glob
import json


PATH_PATH: str = "experiments/moral_foundations/00--Base-Models"

for file in glob.glob(f"{PATH_PATH}/data/**/*.json", recursive=True):
    data = open(file).read()
    data = data.replace("-chat-v1.5-q6_K", "")
    open(file, "w").write(data)
