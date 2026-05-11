import argparse
import os
import tarfile
import re
import json
from multiprocessing import Pool
from tqdm import tqdm
from io import BytesIO

CAPTION_KEYS = [
    "raw_caption",
    "shortIB_captions", "longIB_captions",
    "shortSV_captions", "longSV_captions",
    "shortLLA_captions", "longLLA_captions"
]

def split_caption(text):
    texts = re.split(r'\n|</s>|[.]', text)
    subcap = []
    for text_prompt in texts:
        text_prompt = text_prompt.strip()
        if len(text_prompt) != 0:
            subcap.append(text_prompt)
    return subcap

def process_tar(tar_path):
    tmp_tar_path = tar_path + ".tmp"
    try:
        with tarfile.open(tar_path, 'r') as in_tar, tarfile.open(tmp_tar_path, 'w') as out_tar:
            for member in in_tar.getmembers():
                file_bytes = in_tar.extractfile(member).read()

                if member.name.endswith('.json'):
                    json_obj = json.loads(file_bytes.decode('utf-8'))
                    for k in CAPTION_KEYS:
                        if k in json_obj and isinstance(json_obj[k], str):
                            json_obj[k] = split_caption(json_obj[k])
                    file_bytes = json.dumps(json_obj).encode('utf-8')

                info = tarfile.TarInfo(name=member.name)
                info.size = len(file_bytes)
                out_tar.addfile(info, BytesIO(file_bytes))

        os.replace(tmp_tar_path, tar_path)
        return (tar_path, "success")

    except Exception as e:
        return (tar_path, f"failed: {e}")

def main(shards_dir, num_processes):
    shard_paths = [
        os.path.join(shards_dir, f)
        for f in os.listdir(shards_dir)
        if f.endswith('.tar')
    ]
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap_unordered(process_tar, shard_paths), total=len(shard_paths)))
    for tar, status in results:
        print(f"{tar}: {status}")

if __name__ == "__main__":

    parser = argparse.ArgumentParser(description="Split caption fields inside .json files of WebDataset shards.")
    parser.add_argument("shards_dir", type=str, help="Path to directory containing .tar shards")
    parser.add_argument("num_processes", type=int, help="Number of parallel processes to use")
    args = parser.parse_args()
    main(args.shards_dir, args.num_processes)
